mod models;
mod runtime;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State, WindowEvent};

const DEFAULT_SYSTEM_PROMPT: &str = include_str!("../../runtime/default_system_prompt.txt");

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) client: reqwest::Client,
    pub(crate) cancellations: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    pub(crate) model_download: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    pub(crate) runtime: Arc<Mutex<runtime::RuntimeController>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            cancellations: Arc::new(Mutex::new(HashMap::new())),
            model_download: Arc::new(Mutex::new(None)),
            runtime: Arc::new(Mutex::new(runtime::RuntimeController::default())),
        }
    }
}

fn emit_chat_event(app: &AppHandle, session_id: &str, kind: &str, extra: Value) {
    let mut payload = json!({ "sessionId": session_id, "kind": kind });
    if let (Some(target), Value::Object(fields)) = (payload.as_object_mut(), extra) {
        target.extend(fields);
    }
    let _ = app.emit("chat-event", payload);
}

fn process_sse_frame(app: &AppHandle, session_id: &str, frame: &[u8]) -> bool {
    let frame = String::from_utf8_lossy(frame);
    for line in frame.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            emit_chat_event(app, session_id, "done", json!({}));
            return true;
        }

        let Ok(event) = serde_json::from_str::<Value>(data) else {
            continue;
        };

        if let Some(error) = event.pointer("/error/message").and_then(Value::as_str) {
            emit_chat_event(app, session_id, "error", json!({ "error": error }));
        }

        if let Some(content) = event
            .pointer("/choices/0/delta/content")
            .and_then(Value::as_str)
        {
            emit_chat_event(app, session_id, "delta", json!({ "content": content }));
        }

        if let Some(meta) = event.get("hybrid").filter(|value| value.is_object()) {
            emit_chat_event(app, session_id, "meta", json!({ "meta": meta }));
        }
    }
    false
}

fn remove_cancellation(
    cancellations: &Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    session_id: &str,
    cancellation: &Arc<AtomicBool>,
) {
    if let Ok(mut active) = cancellations.lock() {
        if active
            .get(session_id)
            .is_some_and(|current| Arc::ptr_eq(current, cancellation))
        {
            active.remove(session_id);
        }
    }
}

#[tauri::command]
fn load_sessions(app: AppHandle) -> Result<String, String> {
    let directory = app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?;
    let path = directory.join("sessions.json");
    if !path.exists() {
        return Ok("[]".to_string());
    }
    std::fs::read_to_string(path).map_err(|error| error.to_string())
}

#[tauri::command]
fn save_sessions(app: AppHandle, sessions_json: String) -> Result<(), String> {
    serde_json::from_str::<Value>(&sessions_json).map_err(|error| error.to_string())?;
    let directory = app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?;
    std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let path = directory.join("sessions.json");
    let temporary_path = directory.join("sessions.json.tmp");
    std::fs::write(&temporary_path, sessions_json).map_err(|error| error.to_string())?;
    std::fs::rename(temporary_path, path).map_err(|error| error.to_string())
}

#[tauri::command]
fn stop_chat(state: State<'_, AppState>, session_id: String) {
    if let Ok(active) = state.cancellations.lock() {
        if let Some(cancellation) = active.get(&session_id) {
            cancellation.store(true, Ordering::Relaxed);
        }
    }
}

#[tauri::command]
fn default_system_prompt() -> String {
    DEFAULT_SYSTEM_PROMPT.to_string()
}

#[tauri::command]
fn start_chat(
    app: AppHandle,
    state: State<'_, AppState>,
    session_id: String,
    request_body: String,
) -> Result<(), String> {
    let base_url = runtime::server_base_url(&state)?;
    let request_body: Value =
        serde_json::from_str(&request_body).map_err(|error| error.to_string())?;
    let request_url = format!("{base_url}/chat/completions");
    let client = state.client.clone();
    let cancellations = state.cancellations.clone();
    let cancellation = Arc::new(AtomicBool::new(false));
    if let Ok(mut active) = cancellations.lock() {
        active.insert(session_id.clone(), cancellation.clone());
    }

    tauri::async_runtime::spawn(async move {
        let result = client.post(&request_url).json(&request_body).send().await;
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                emit_chat_event(
                    &app,
                    &session_id,
                    "error",
                    json!({ "error": format!("Could not reach HybridAgent: {error}") }),
                );
                remove_cancellation(&cancellations, &session_id, &cancellation);
                return;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let detail = response.text().await.unwrap_or_default();
            emit_chat_event(
                &app,
                &session_id,
                "error",
                json!({ "error": format!("HybridAgent returned {status}: {detail}") }),
            );
            remove_cancellation(&cancellations, &session_id, &cancellation);
            return;
        }

        let mut stream = response.bytes_stream();
        let mut buffer = Vec::<u8>::new();
        let mut finished = false;

        while let Some(chunk) = stream.next().await {
            if cancellation.load(Ordering::Relaxed) {
                emit_chat_event(&app, &session_id, "stopped", json!({}));
                finished = true;
                break;
            }

            match chunk {
                Ok(bytes) => buffer.extend_from_slice(&bytes),
                Err(error) => {
                    emit_chat_event(
                        &app,
                        &session_id,
                        "error",
                        json!({ "error": format!("Stream interrupted: {error}") }),
                    );
                    finished = true;
                    break;
                }
            }

            while let Some(boundary) = buffer.windows(2).position(|window| window == b"\n\n") {
                let frame: Vec<u8> = buffer.drain(..boundary + 2).collect();
                if process_sse_frame(&app, &session_id, &frame) {
                    finished = true;
                    break;
                }
            }
            if finished {
                break;
            }
        }

        if !finished && !buffer.is_empty() {
            finished = process_sse_frame(&app, &session_id, &buffer);
        }
        if !finished {
            emit_chat_event(&app, &session_id, "done", json!({}));
        }
        remove_cancellation(&cancellations, &session_id, &cancellation);
    });

    Ok(())
}

fn main() {
    let application = tauri::Builder::default()
        .manage(AppState::new())
        .on_window_event(|window, event| {
            if matches!(event, WindowEvent::CloseRequested { .. }) {
                runtime::stop_managed_server(window.state::<AppState>().inner());
            }
        })
        .invoke_handler(tauri::generate_handler![
            default_system_prompt,
            load_sessions,
            save_sessions,
            start_chat,
            stop_chat,
            runtime::load_model,
            runtime::unload_model,
            runtime::engine_status,
            models::model_catalog,
            models::start_model_download,
            models::cancel_model_download
        ])
        .build(tauri::generate_context!())
        .expect("error while running HybridChat");
    application.run(|app_handle, event| {
        if matches!(event, RunEvent::Exit | RunEvent::ExitRequested { .. }) {
            runtime::stop_managed_server(app_handle.state::<AppState>().inner());
        }
    });
}
