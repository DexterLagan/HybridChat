use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use futures_util::StreamExt;
use reqwest::header::RANGE;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager, State};

use crate::runtime;
use crate::AppState;

const RUNTIME_DISK_RESERVE: u64 = 1024 * 1024 * 1024;
const PROGRESS_INTERVAL: u64 = 4 * 1024 * 1024;
const QUANT_BITS: u8 = 6;

#[derive(Clone, Copy)]
struct RemoteFile {
    name: &'static str,
    bytes: u64,
    sha256: Option<&'static str>,
}

#[derive(Clone, Copy)]
struct RemoteModel {
    id: &'static str,
    label: &'static str,
    repo: &'static str,
    revision: &'static str,
    quantized_bytes: u64,
    files: &'static [RemoteFile],
}

const MODEL_8B_FILES: &[RemoteFile] = &[
    RemoteFile {
        name: "adapter.pt",
        bytes: 67_161_229,
        sha256: Some("e5e91f8c61e68be9bd858d24816722eab110e3d1a9febed64481d7ed8c7ceca5"),
    },
    RemoteFile {
        name: "adapter_config.json",
        bytes: 38,
        sha256: None,
    },
    RemoteFile {
        name: "chat_template.jinja",
        bytes: 4_116,
        sha256: None,
    },
    RemoteFile {
        name: "config.json",
        bytes: 1_595,
        sha256: None,
    },
    RemoteFile {
        name: "generation_config.json",
        bytes: 139,
        sha256: None,
    },
    RemoteFile {
        name: "model.safetensors",
        bytes: 16_377_175_448,
        sha256: Some("2de074d380f4d7f82812245b8d3fefb075936366136f60d03c57dcc75f4d2c1a"),
    },
    RemoteFile {
        name: "tokenizer.json",
        bytes: 11_423_021,
        sha256: Some("e3d0e3a96786f3ad1e667f478fc7d32ef83ca61bd506a3e11b912c88508296aa"),
    },
    RemoteFile {
        name: "tokenizer_config.json",
        bytes: 445,
        sha256: None,
    },
];

const MODEL_14B_FILES: &[RemoteFile] = &[
    RemoteFile {
        name: "adapter.pt",
        bytes: 104_922_253,
        sha256: Some("635118c9790755a4e9baff6d9025790edd2b73e8cfe6466504101e56dd994844"),
    },
    RemoteFile {
        name: "adapter_config.json",
        bytes: 38,
        sha256: None,
    },
    RemoteFile {
        name: "chat_template.jinja",
        bytes: 4_116,
        sha256: None,
    },
    RemoteFile {
        name: "config.json",
        bytes: 1_683,
        sha256: None,
    },
    RemoteFile {
        name: "generation_config.json",
        bytes: 139,
        sha256: None,
    },
    RemoteFile {
        name: "model.safetensors",
        bytes: 29_531_239_072,
        sha256: Some("478cc3a0855ef3a169bee6768c6fa8478477e440ef9364854a623a65d723d5e5"),
    },
    RemoteFile {
        name: "tokenizer.json",
        bytes: 11_423_021,
        sha256: Some("e3d0e3a96786f3ad1e667f478fc7d32ef83ca61bd506a3e11b912c88508296aa"),
    },
    RemoteFile {
        name: "tokenizer_config.json",
        bytes: 445,
        sha256: None,
    },
];

const MODELS: &[RemoteModel] = &[
    RemoteModel {
        id: "HybridAgent-8B-Agentic",
        label: "HybridAgent 8B Agentic",
        repo: "togethercomputer/HybridAgent-8B-Agentic",
        revision: "13a5c61b1880d7bcc2fac2e28d478809722beb8f",
        quantized_bytes: 6_732_346_955,
        files: MODEL_8B_FILES,
    },
    RemoteModel {
        id: "HybridAgent-14B-Agentic",
        label: "HybridAgent 14B Agentic",
        repo: "togethercomputer/HybridAgent-14B-Agentic",
        revision: "1b726e326a4bfd38dc3c2251451f6fe0106d0d1c",
        quantized_bytes: 12_114_102_910,
        files: MODEL_14B_FILES,
    },
];

fn model(id: &str) -> Option<&'static RemoteModel> {
    MODELS.iter().find(|model| model.id == id)
}

pub fn models_directory(app: &AppHandle) -> Result<PathBuf, String> {
    let directory = app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("models");
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    Ok(directory)
}

fn development_models_directory() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("HYBRIDCHAT_SOURCE_MODELS_DIR") {
        return Some(PathBuf::from(path));
    }
    #[cfg(debug_assertions)]
    {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        return Some(
            manifest
                .join("../../models")
                .canonicalize()
                .unwrap_or_else(|_| manifest.join("../../models")),
        );
    }
    #[cfg(not(debug_assertions))]
    {
        None
    }
}

fn huggingface_cached_source(model: &RemoteModel) -> Option<PathBuf> {
    let hub_cache = if let Some(path) = std::env::var_os("HF_HUB_CACHE") {
        PathBuf::from(path)
    } else if let Some(home) = std::env::var_os("HF_HOME") {
        PathBuf::from(home).join("hub")
    } else {
        PathBuf::from(std::env::var_os("HOME")?).join(".cache/huggingface/hub")
    };
    let repo = model.repo.replace('/', "--");
    let snapshot = hub_cache
        .join(format!("models--{repo}"))
        .join("snapshots")
        .join(model.revision);
    model
        .files
        .iter()
        .all(|file| file_is_complete(&snapshot, file))
        .then_some(snapshot)
}

fn is_quantized_model(directory: &Path) -> bool {
    if !directory.join("adapter.pt").is_file()
        || !directory.join("model.safetensors.index.json").is_file()
    {
        return false;
    }
    let Ok(metadata) = fs::read_to_string(directory.join("hybridchat_model.json")) else {
        return false;
    };
    serde_json::from_str::<Value>(&metadata)
        .ok()
        .and_then(|value| value.get("weight_bits").and_then(Value::as_u64))
        == Some(QUANT_BITS as u64)
}

pub fn installed_model_directory(
    app: &AppHandle,
    model_id: &str,
) -> Result<Option<PathBuf>, String> {
    if model(model_id).is_none() {
        return Err(format!("Unknown model {model_id}"));
    }
    let app_data_model = models_directory(app)?.join(model_id);
    if is_quantized_model(&app_data_model) {
        return Ok(Some(app_data_model));
    }
    if std::env::var_os("HYBRIDCHAT_DISABLE_DEV_QUANT_MODELS").is_none() {
        if let Some(development_models) = development_models_directory() {
            let development_model = development_models.join(format!("{model_id}-MLX-6bit"));
            if is_quantized_model(&development_model) {
                return Ok(Some(development_model));
            }
        }
    }
    Ok(None)
}

pub fn source_model_directory(app: &AppHandle, model_id: &str) -> Result<Option<PathBuf>, String> {
    let model = model(model_id).ok_or_else(|| format!("Unknown model {model_id}"))?;
    let app_source = models_directory(app)?.join(".source").join(model.id);
    if model
        .files
        .iter()
        .all(|file| file_is_complete(&app_source, file))
    {
        return Ok(Some(app_source));
    }
    if let Some(development_models) = development_models_directory() {
        let development_source = development_models.join(model.id);
        if model
            .files
            .iter()
            .all(|file| file_is_complete(&development_source, file))
        {
            return Ok(Some(development_source));
        }
    }
    if let Some(cached_source) = huggingface_cached_source(model) {
        return Ok(Some(cached_source));
    }
    Ok(None)
}

fn app_source_directory(root: &Path, model: &RemoteModel) -> PathBuf {
    root.join(".source").join(model.id)
}

fn file_is_complete(directory: &Path, file: &RemoteFile) -> bool {
    fs::metadata(directory.join(file.name))
        .map(|metadata| metadata.is_file() && metadata.len() == file.bytes)
        .unwrap_or(false)
}

fn partial_size(directory: &Path, file: &RemoteFile) -> u64 {
    let partial = directory.join(format!("{}.part", file.name));
    fs::metadata(partial)
        .map(|metadata| metadata.len().min(file.bytes))
        .unwrap_or(0)
}

fn model_bytes(model: &RemoteModel) -> u64 {
    model.files.iter().map(|file| file.bytes).sum()
}

fn partial_source_bytes(root: &Path, model: &RemoteModel) -> u64 {
    let directory = app_source_directory(root, model);
    model
        .files
        .iter()
        .map(|file| {
            if file_is_complete(&directory, file) {
                file.bytes
            } else {
                partial_size(&directory, file)
            }
        })
        .sum()
}

#[tauri::command]
pub fn model_catalog(app: AppHandle) -> Result<Value, String> {
    let root = models_directory(&app)?;
    let available = fs2::available_space(&root).map_err(|error| error.to_string())?;
    let records: Vec<Value> = MODELS
        .iter()
        .map(|model| {
            let installed = installed_model_directory(&app, model.id)
                .ok()
                .flatten()
                .is_some();
            let source_directory = source_model_directory(&app, model.id).ok().flatten();
            let partial_directory = app_source_directory(&root, model);
            let downloaded_bytes = if source_directory.is_some() {
                model_bytes(model)
            } else {
                model
                    .files
                    .iter()
                    .map(|file| partial_size(&partial_directory, file))
                    .sum::<u64>()
            };
            let remaining_bytes = if source_directory.is_some() {
                0
            } else {
                model_bytes(model).saturating_sub(downloaded_bytes)
            };
            json!({
                "id": model.id,
                "label": model.label,
                "repoId": model.repo,
                "revision": model.revision,
                "sourceBytes": model_bytes(model),
                "quantizedBytes": model.quantized_bytes,
                "quantization": "MLX 6-bit affine",
                "downloadedBytes": downloaded_bytes,
                "remainingBytes": remaining_bytes,
                "sourceAvailable": source_directory.is_some(),
                "sourcePathKind": if source_directory.is_some() { "local" } else { "download" },
                "installed": installed,
            })
        })
        .collect();

    Ok(json!({
        "models": records,
        "modelsDirectory": root,
        "availableBytes": available,
        "runtimeReserveBytes": RUNTIME_DISK_RESERVE,
    }))
}

#[tauri::command]
pub fn start_model_download(
    app: AppHandle,
    state: State<'_, AppState>,
    model_ids: Vec<String>,
) -> Result<(), String> {
    if model_ids.is_empty() {
        return Err("Select at least one model to download".to_string());
    }
    let mut selected: Vec<&'static RemoteModel> = Vec::new();
    for id in &model_ids {
        let Some(model) = model(id) else {
            return Err(format!("Unknown model {id}"));
        };
        if !selected.iter().any(|current| current.id == model.id) {
            selected.push(model);
        }
    }
    selected.sort_by_key(|model| std::cmp::Reverse(model_bytes(model)));

    let root = models_directory(&app)?;
    let mut retained_output = 0u64;
    let mut peak_staging = 0u64;
    let mut any_new = false;
    for model in &selected {
        if installed_model_directory(&app, model.id)?.is_some() {
            continue;
        }
        any_new = true;
        let source_missing = if source_model_directory(&app, model.id)?.is_some() {
            0
        } else {
            model_bytes(model).saturating_sub(partial_source_bytes(&root, model))
        };
        peak_staging = peak_staging.max(
            retained_output
                .saturating_add(source_missing)
                .saturating_add(model.quantized_bytes),
        );
        retained_output = retained_output.saturating_add(model.quantized_bytes);
    }
    let available = fs2::available_space(&root).map_err(|error| error.to_string())?;
    let reserved = if any_new { RUNTIME_DISK_RESERVE } else { 0 };
    if available < peak_staging.saturating_add(reserved) {
        return Err(format!(
            "Not enough free space. The selected quantized models need up to {} of temporary model space and the runtime reserve is {}; {} is available on the models drive.",
            format_bytes(peak_staging),
            format_bytes(reserved),
            format_bytes(available),
        ));
    }

    let cancellation = Arc::new(AtomicBool::new(false));
    {
        let mut active = state
            .model_download
            .lock()
            .map_err(|_| "Download state unavailable".to_string())?;
        if active
            .as_ref()
            .is_some_and(|flag| !flag.load(Ordering::Relaxed))
        {
            return Err("A model download is already in progress".to_string());
        }
        *active = Some(cancellation.clone());
    }

    let app_handle = app.clone();
    let client = state.client.clone();
    let control = state.model_download.clone();
    let selected_ids = selected
        .iter()
        .map(|model| model.id.to_string())
        .collect::<Vec<_>>();
    tauri::async_runtime::spawn(async move {
        let result = run_downloads(
            &app_handle,
            &client,
            &root,
            selected,
            &selected_ids,
            &cancellation,
        )
        .await;

        match result {
            Ok(()) if cancellation.load(Ordering::Relaxed) => {
                emit(&app_handle, "cancelled", json!({}));
            }
            Ok(()) => emit(&app_handle, "complete", json!({})),
            Err(error) => emit(&app_handle, "error", json!({ "message": error })),
        }

        if let Ok(mut active) = control.lock() {
            if active
                .as_ref()
                .is_some_and(|flag| Arc::ptr_eq(flag, &cancellation))
            {
                *active = None;
            }
        }
    });
    Ok(())
}

#[tauri::command]
pub fn cancel_model_download(state: State<'_, AppState>) {
    if let Ok(active) = state.model_download.lock() {
        if let Some(cancellation) = active.as_ref() {
            cancellation.store(true, Ordering::Relaxed);
        }
    }
}

async fn run_downloads(
    app: &AppHandle,
    client: &reqwest::Client,
    root: &Path,
    selected: Vec<&'static RemoteModel>,
    selected_ids: &[String],
    cancellation: &AtomicBool,
) -> Result<(), String> {
    let app_for_setup = app.clone();
    emit(
        app,
        "runtime-setup",
        json!({ "message": "Preparing the local MLX runtime" }),
    );
    let environment =
        tauri::async_runtime::spawn_blocking(move || runtime::prepare_engine(&app_for_setup))
            .await
            .map_err(|error| error.to_string())??;
    let total_bytes = selected.iter().map(|model| model_bytes(model)).sum::<u64>();
    let downloaded = Arc::new(AtomicU64::new(
        selected
            .iter()
            .map(|model| {
                if installed_model_directory(app, model.id)
                    .ok()
                    .flatten()
                    .is_some()
                    || source_model_directory(app, model.id)
                        .ok()
                        .flatten()
                        .is_some()
                {
                    model_bytes(model)
                } else {
                    partial_source_bytes(root, model)
                }
            })
            .sum(),
    ));

    emit(
        app,
        "started",
        json!({
            "modelIds": selected_ids,
            "totalBytes": total_bytes,
            "downloadedBytes": downloaded.load(Ordering::Relaxed),
        }),
    );

    for model in selected {
        if cancellation.load(Ordering::Relaxed) {
            return Ok(());
        }
        if let Some(installed) = installed_model_directory(app, model.id)? {
            emit(
                app,
                "model-complete",
                json!({ "modelId": model.id, "installed": true, "path": installed }),
            );
            continue;
        }

        let source_directory = source_model_directory(app, model.id)?;
        let source_directory = if let Some(source) = source_directory {
            source
        } else {
            let directory = app_source_directory(root, model);
            fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
            for file in model.files {
                if cancellation.load(Ordering::Relaxed) {
                    return Ok(());
                }
                if file_is_complete(&directory, file) {
                    continue;
                }
                download_file(
                    app,
                    client,
                    model,
                    file,
                    &directory,
                    &downloaded,
                    total_bytes,
                    cancellation,
                )
                .await?;
            }
            directory
        };

        emit(
            app,
            "quantizing",
            json!({ "modelId": model.id, "bits": QUANT_BITS, "message": "Quantizing to MLX 6-bit" }),
        );
        let app_for_conversion = app.clone();
        let source_for_conversion = source_directory.clone();
        let output_for_conversion = root.join(model.id);
        let environment_for_conversion = environment.clone();
        tauri::async_runtime::spawn_blocking(move || {
            runtime::run_conversion(
                &app_for_conversion,
                &environment_for_conversion,
                &source_for_conversion,
                &output_for_conversion,
            )
        })
        .await
        .map_err(|error| error.to_string())??;

        let downloaded_source = source_directory.starts_with(root.join(".source"));
        if downloaded_source {
            fs::remove_dir_all(&source_directory).map_err(|error| error.to_string())?;
        }
        emit(
            app,
            "model-complete",
            json!({ "modelId": model.id, "installed": true, "quantizedBytes": model.quantized_bytes }),
        );
    }
    Ok(())
}

async fn download_file(
    app: &AppHandle,
    client: &reqwest::Client,
    model: &RemoteModel,
    file: &RemoteFile,
    directory: &Path,
    downloaded: &AtomicU64,
    total_bytes: u64,
    cancellation: &AtomicBool,
) -> Result<(), String> {
    let final_path = directory.join(file.name);
    let partial_path = directory.join(format!("{}.part", file.name));
    let mut existing = fs::metadata(&partial_path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    if existing > file.bytes {
        fs::remove_file(&partial_path).map_err(|error| error.to_string())?;
        downloaded.fetch_sub(existing.min(file.bytes), Ordering::Relaxed);
        existing = 0;
    }

    if existing == file.bytes {
        if let Some(expected_hash) = file.sha256 {
            if hash_file(&partial_path)? != expected_hash {
                fs::remove_file(&partial_path).map_err(|error| error.to_string())?;
                downloaded.fetch_sub(file.bytes, Ordering::Relaxed);
                existing = 0;
            }
        }
        if existing == file.bytes {
            fs::rename(&partial_path, &final_path).map_err(|error| error.to_string())?;
            return Ok(());
        }
    }

    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}?download=true",
        model.repo, model.revision, file.name
    );
    let mut request = client.get(url);
    if existing > 0 {
        request = request.header(RANGE, format!("bytes={existing}-"));
    }
    let response = request.send().await.map_err(|error| error.to_string())?;
    let status = response.status();
    let append = status == reqwest::StatusCode::PARTIAL_CONTENT;
    if !status.is_success() {
        return Err(format!(
            "Hugging Face download failed for {} (HTTP {})",
            file.name, status
        ));
    }

    if existing > 0 && !append {
        downloaded.fetch_sub(existing, Ordering::Relaxed);
        existing = 0;
    }

    let mut options = OpenOptions::new();
    options.create(true).write(true);
    if append && existing > 0 {
        options.append(true);
    } else {
        options.truncate(true);
    }
    let mut output = options
        .open(&partial_path)
        .map_err(|error| error.to_string())?;

    let mut hasher = if file.sha256.is_some() {
        let mut hasher = Sha256::new();
        if append && existing > 0 {
            let mut previous = File::open(&partial_path).map_err(|error| error.to_string())?;
            let mut bytes = [0u8; 1024 * 1024];
            loop {
                let read = previous
                    .read(&mut bytes)
                    .map_err(|error| error.to_string())?;
                if read == 0 {
                    break;
                }
                hasher.update(&bytes[..read]);
            }
        }
        Some(hasher)
    } else {
        None
    };

    let mut bytes_since_progress = 0u64;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        if cancellation.load(Ordering::Relaxed) {
            output.flush().map_err(|error| error.to_string())?;
            output.sync_all().map_err(|error| error.to_string())?;
            return Ok(());
        }
        let chunk = chunk.map_err(|error| error.to_string())?;
        output
            .write_all(&chunk)
            .map_err(|error| error.to_string())?;
        if let Some(hasher) = hasher.as_mut() {
            hasher.update(&chunk);
        }
        let count = chunk.len() as u64;
        downloaded.fetch_add(count, Ordering::Relaxed);
        bytes_since_progress += count;
        if bytes_since_progress >= PROGRESS_INTERVAL {
            bytes_since_progress = 0;
            emit(
                app,
                "progress",
                json!({
                    "modelId": model.id,
                    "fileName": file.name,
                    "downloadedBytes": downloaded.load(Ordering::Relaxed),
                    "totalBytes": total_bytes,
                }),
            );
        }
    }
    output.flush().map_err(|error| error.to_string())?;
    output.sync_all().map_err(|error| error.to_string())?;
    drop(output);

    let actual_size = fs::metadata(&partial_path)
        .map_err(|error| error.to_string())?
        .len();
    if actual_size != file.bytes {
        return Err(format!(
            "{} downloaded incompletely: expected {}, got {}. Retry to resume.",
            file.name, file.bytes, actual_size
        ));
    }

    if let (Some(expected_hash), Some(hasher)) = (file.sha256, hasher) {
        let actual_hash = format!("{:x}", hasher.finalize());
        if actual_hash != expected_hash {
            fs::remove_file(&partial_path).map_err(|error| error.to_string())?;
            downloaded.fetch_sub(file.bytes, Ordering::Relaxed);
            return Err(format!("Checksum validation failed for {}", file.name));
        }
    }

    fs::rename(partial_path, final_path).map_err(|error| error.to_string())?;
    emit(
        app,
        "progress",
        json!({
            "modelId": model.id,
            "fileName": file.name,
            "downloadedBytes": downloaded.load(Ordering::Relaxed),
            "totalBytes": total_bytes,
        }),
    );
    Ok(())
}

fn hash_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    let mut bytes = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut bytes).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        hasher.update(&bytes[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn emit(app: &AppHandle, kind: &str, extra: Value) {
    let mut payload = json!({ "kind": kind });
    if let (Some(target), Value::Object(fields)) = (payload.as_object_mut(), extra) {
        target.extend(fields);
    }
    let _ = app.emit("model-event", payload);
}

pub fn format_bytes(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
}
