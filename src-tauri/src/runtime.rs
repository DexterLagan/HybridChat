use std::fs::{self, OpenOptions};
use std::io::Cursor;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use flate2::read::GzDecoder;
use reqwest::blocking::Client as BlockingClient;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager, State};

use crate::models;
use crate::AppState;

const MLX_ENGINE_SOURCE: &str = include_str!("../../runtime/mlx_engine.py");
const MLX_SERVER_SOURCE: &str = include_str!("../../runtime/server.py");
const CONVERTER_SOURCE: &str = include_str!("../../runtime/convert_model.py");
const BOOTSTRAP_SOURCE: &str = include_str!("../../runtime/bootstrap.py");
const REQUIREMENTS_SOURCE: &str = include_str!("../../runtime/requirements.txt");

#[derive(Clone)]
pub(crate) struct RuntimeEnvironment {
    pub python: PathBuf,
    pub scripts_dir: PathBuf,
    pub engine_source: PathBuf,
}

#[derive(Default)]
pub(crate) struct RuntimeController {
    child: Option<Child>,
    base_url: Option<String>,
}

impl RuntimeController {
    fn base_url(&self) -> Option<String> {
        self.base_url.clone()
    }

    fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.base_url = None;
    }
}

impl Drop for RuntimeController {
    fn drop(&mut self) {
        self.stop();
    }
}

pub(crate) fn stop_managed_server(state: &AppState) {
    if let Ok(mut runtime) = state.runtime.lock() {
        runtime.stop();
    }
}

fn emit_runtime_event(app: &AppHandle, kind: &str, message: &str) {
    let _ = app.emit("runtime-event", json!({ "kind": kind, "message": message }));
}

fn app_runtime_dir(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("runtime"))
}

fn write_runtime_sources(directory: &Path) -> Result<(), String> {
    fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    for (name, content) in [
        ("mlx_engine.py", MLX_ENGINE_SOURCE),
        ("server.py", MLX_SERVER_SOURCE),
        ("convert_model.py", CONVERTER_SOURCE),
        ("bootstrap.py", BOOTSTRAP_SOURCE),
        ("requirements.txt", REQUIREMENTS_SOURCE),
    ] {
        fs::write(directory.join(name), content).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn command_succeeds(program: &Path, args: &[&str], current_dir: Option<&Path>) -> bool {
    let mut command = Command::new(program);
    command
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(directory) = current_dir {
        command.current_dir(directory);
    }
    command
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn compatible_python(python: &Path, current_dir: Option<&Path>) -> bool {
    command_succeeds(
        python,
        &[
            "-c",
            "import sys, mlx, mlx_lm, torch, fastapi, uvicorn; assert sys.version_info >= (3, 11)",
        ],
        current_dir,
    )
}

fn find_uv() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("UV_BIN") {
        let candidate = PathBuf::from(path);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        for directory in std::env::split_paths(&path) {
            let candidate = directory.join("uv");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let mut candidates = vec![
        PathBuf::from("/opt/homebrew/bin/uv"),
        PathBuf::from("/usr/local/bin/uv"),
    ];
    if let Some(home) = home {
        candidates.push(home.join(".local/bin/uv"));
        candidates.push(home.join(".cargo/bin/uv"));
    }
    candidates.into_iter().find(|path| path.is_file())
}

fn download_uv(app: &AppHandle, tools_dir: &Path) -> Result<PathBuf, String> {
    #[cfg(not(target_os = "macos"))]
    return Err(
        "HybridChat's managed MLX engine currently requires macOS on Apple Silicon".to_string(),
    );

    #[cfg(target_os = "macos")]
    {
        let arch = match std::env::consts::ARCH {
            "aarch64" => "aarch64",
            "x86_64" => "x86_64",
            other => return Err(format!("Unsupported Mac architecture for MLX: {other}")),
        };
        fs::create_dir_all(tools_dir).map_err(|error| error.to_string())?;
        let uv_path = tools_dir.join("uv");
        if uv_path.is_file() {
            return Ok(uv_path);
        }
        emit_runtime_event(app, "progress", "Downloading the small uv Python manager");
        let url = format!(
            "https://github.com/astral-sh/uv/releases/latest/download/uv-{arch}-apple-darwin.tar.gz"
        );
        let response = BlockingClient::builder()
            .user_agent("HybridChat")
            .build()
            .map_err(|error| error.to_string())?
            .get(url)
            .send()
            .map_err(|error| error.to_string())?
            .error_for_status()
            .map_err(|error| error.to_string())?;
        let bytes = response.bytes().map_err(|error| error.to_string())?;
        let decoder = GzDecoder::new(Cursor::new(bytes));
        let mut archive = tar::Archive::new(decoder);
        let mut found = false;
        for entry in archive.entries().map_err(|error| error.to_string())? {
            let mut entry = entry.map_err(|error| error.to_string())?;
            let is_uv = entry
                .path()
                .map_err(|error| error.to_string())?
                .file_name()
                .is_some_and(|name| name == "uv");
            if is_uv {
                entry.unpack(&uv_path).map_err(|error| error.to_string())?;
                found = true;
                break;
            }
        }
        if !found {
            return Err("The uv archive did not contain its uv executable".to_string());
        }
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&uv_path, fs::Permissions::from_mode(0o755))
            .map_err(|error| error.to_string())?;
        Ok(uv_path)
    }
}

fn run_logged(
    app: &AppHandle,
    command: &mut Command,
    log_path: &Path,
    stage: &str,
    message: &str,
) -> Result<(), String> {
    emit_runtime_event(app, "progress", message);
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|error| error.to_string())?;
    let stdout = log.try_clone().map_err(|error| error.to_string())?;
    let status = command
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(log))
        .status()
        .map_err(|error| format!("Could not {stage}: {error}"))?;
    if !status.success() {
        return Err(format!(
            "Could not {stage} (status {status}). See {}",
            log_path.display()
        ));
    }
    Ok(())
}

pub(crate) fn prepare_engine(app: &AppHandle) -> Result<RuntimeEnvironment, String> {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Err(
            "HybridChat's MLX inference engine currently requires Apple Silicon macOS".to_string(),
        );
    }
    let scripts_dir = app_runtime_dir(app)?;
    write_runtime_sources(&scripts_dir)?;

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dev_root = manifest_dir
        .join("../..")
        .canonicalize()
        .unwrap_or(manifest_dir.clone());
    let dev_python = dev_root.join(".venv/bin/python");
    let force_managed_runtime = std::env::var_os("HYBRIDCHAT_FORCE_MANAGED_RUNTIME").is_some();
    if !force_managed_runtime
        && dev_python.is_file()
        && dev_root.join("hybrid_agent/__init__.py").is_file()
        && compatible_python(&dev_python, Some(&dev_root))
    {
        return Ok(RuntimeEnvironment {
            python: dev_python,
            scripts_dir,
            engine_source: dev_root,
        });
    }

    let venv_dir = scripts_dir.join("venv");
    let python = venv_dir.join("bin/python");
    let engine_source = scripts_dir.join("engine-source");
    if python.is_file() && compatible_python(&python, Some(&scripts_dir)) {
        if !engine_source.join("hybrid_agent/__init__.py").is_file() {
            run_logged(
                app,
                Command::new(&python)
                    .arg(scripts_dir.join("bootstrap.py"))
                    .arg("--source-dir")
                    .arg(&engine_source),
                &scripts_dir.join("setup.log"),
                "prepare the HybridAgent engine source",
                "Downloading the HybridAgent inference code",
            )?;
        }
        return Ok(RuntimeEnvironment {
            python,
            scripts_dir,
            engine_source,
        });
    }

    let tools_dir = scripts_dir.join("tools");
    let uv = match find_uv() {
        Some(path) => path,
        None => download_uv(app, &tools_dir)?,
    };
    let python_install_dir = scripts_dir.join("python");
    let log_path = scripts_dir.join("setup.log");
    run_logged(
        app,
        Command::new(&uv)
            .arg("python")
            .arg("install")
            .arg("3.11")
            .arg("--install-dir")
            .arg(&python_install_dir)
            .env("UV_PYTHON_INSTALL_DIR", &python_install_dir),
        &log_path,
        "install Python 3.11",
        "Installing a private Python 3.11 runtime",
    )?;
    if venv_dir.exists() {
        fs::remove_dir_all(&venv_dir).map_err(|error| error.to_string())?;
    }
    let mut venv_command = Command::new(&uv);
    venv_command
        .arg("venv")
        .arg("--python")
        .arg("3.11")
        .arg("--managed-python")
        .arg("--seed")
        .arg(&venv_dir)
        .env("UV_PYTHON_INSTALL_DIR", &python_install_dir);
    run_logged(
        app,
        &mut venv_command,
        &log_path,
        "create the private Python environment",
        "Creating the local inference environment",
    )?;
    let mut install = Command::new(&uv);
    install
        .arg("pip")
        .arg("install")
        .arg("--python")
        .arg(&python)
        .arg("--requirement")
        .arg(scripts_dir.join("requirements.txt"));
    run_logged(
        app,
        &mut install,
        &log_path,
        "install MLX and HybridAgent dependencies",
        "Installing MLX, PyTorch adapter support, and the HybridAgent service",
    )?;
    run_logged(
        app,
        Command::new(&python)
            .arg(scripts_dir.join("bootstrap.py"))
            .arg("--source-dir")
            .arg(&engine_source),
        &log_path,
        "download the HybridAgent inference code",
        "Preparing the HybridAgent inference engine",
    )?;

    if !compatible_python(&python, Some(&scripts_dir)) {
        return Err(
            "The managed Python environment is missing required MLX dependencies".to_string(),
        );
    }
    Ok(RuntimeEnvironment {
        python,
        scripts_dir,
        engine_source,
    })
}

pub(crate) fn run_conversion(
    app: &AppHandle,
    environment: &RuntimeEnvironment,
    source: &Path,
    output: &Path,
) -> Result<(), String> {
    let log_path = environment.scripts_dir.join("conversion.log");
    let mut command = Command::new(&environment.python);
    command
        .arg(environment.scripts_dir.join("convert_model.py"))
        .arg("--source")
        .arg(source)
        .arg("--output")
        .arg(output)
        .arg("--bits")
        .arg("6")
        .arg("--group-size")
        .arg("64")
        .current_dir(&environment.scripts_dir);
    run_logged(
        app,
        &mut command,
        &log_path,
        "quantize the HybridAgent checkpoint",
        "Converting weights to MLX 6-bit format and preserving HybridLink",
    )
}

fn server_command(environment: &RuntimeEnvironment, models_dir: &Path, port: u16) -> Command {
    let mut command = Command::new(&environment.python);
    command
        .arg(environment.scripts_dir.join("server.py"))
        .arg("--models-dir")
        .arg(models_dir)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .current_dir(&environment.scripts_dir)
        .env("PYTHONPATH", &environment.engine_source)
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null());
    command
}

fn ensure_server(
    app: &AppHandle,
    runtime: &Arc<Mutex<RuntimeController>>,
    client: &reqwest::Client,
) -> Result<String, String> {
    let mut controller = runtime
        .lock()
        .map_err(|_| "Runtime state unavailable".to_string())?;
    if let Some(child) = controller.child.as_mut() {
        if child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_none()
        {
            if let Some(base_url) = controller.base_url.clone() {
                return Ok(base_url);
            }
        }
        controller.stop();
    }

    let environment = prepare_engine(app)?;
    let models_dir = models::models_directory(app)?;
    let preferred = TcpListener::bind("127.0.0.1:8100");
    let port = match preferred {
        Ok(listener) => {
            let port = listener
                .local_addr()
                .map_err(|error| error.to_string())?
                .port();
            drop(listener);
            port
        }
        Err(_) => TcpListener::bind("127.0.0.1:0")
            .map_err(|error| error.to_string())?
            .local_addr()
            .map_err(|error| error.to_string())?
            .port(),
    };
    let app_data = app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?;
    let log_path = app_data.join("hybridchat-engine.log");
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|error| error.to_string())?;
    let stdout = log.try_clone().map_err(|error| error.to_string())?;
    let mut child = server_command(&environment, &models_dir, port)
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(log))
        .spawn()
        .map_err(|error| format!("Could not start HybridChat's MLX engine: {error}"))?;

    let base_url = format!("http://127.0.0.1:{port}/v1");
    let health_url = format!("http://127.0.0.1:{port}/health");
    let blocking_client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|error| error.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            return Err(format!(
                "HybridChat's MLX engine exited during startup ({status}). See {}",
                log_path.display()
            ));
        }
        if blocking_client.get(&health_url).send().is_ok() {
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "Timed out starting the MLX engine. See {}",
                log_path.display()
            ));
        }
        thread::sleep(Duration::from_millis(350));
    }

    controller.child = Some(child);
    controller.base_url = Some(base_url.clone());
    let _ = client;
    emit_runtime_event(app, "ready", "Local MLX engine is ready");
    Ok(base_url)
}

#[tauri::command]
pub async fn load_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model_id: String,
) -> Result<Value, String> {
    let model_path = models::installed_model_directory(&app, &model_id)?
        .ok_or_else(|| format!("{model_id} is not installed yet"))?;
    let runtime = state.runtime.clone();
    let client = state.client.clone();
    let app_for_start = app.clone();
    let server_url = tauri::async_runtime::spawn_blocking(move || {
        ensure_server(&app_for_start, &runtime, &client)
    })
    .await
    .map_err(|error| error.to_string())??;

    let server_root = server_url.trim_end_matches("/v1");
    let response = state
        .client
        .post(format!("{server_root}/admin/models/load"))
        .json(&json!({ "model_id": model_id, "model_path": model_path }))
        .send()
        .await
        .map_err(|error| error.to_string())?;
    let status = response.status();
    let body = response
        .json::<Value>()
        .await
        .map_err(|error| error.to_string())?;
    if !status.is_success() {
        return Err(body
            .get("detail")
            .and_then(Value::as_str)
            .unwrap_or("Model load failed")
            .to_string());
    }
    Ok(body)
}

#[tauri::command]
pub async fn unload_model(state: State<'_, AppState>) -> Result<Value, String> {
    let base_url = state
        .runtime
        .lock()
        .map_err(|_| "Runtime state unavailable".to_string())?
        .base_url();
    let Some(base_url) = base_url else {
        return Ok(json!({ "ok": true, "loaded": false, "model": null, "device": "mps" }));
    };
    let server_root = base_url.trim_end_matches("/v1");
    let response = state
        .client
        .post(format!("{server_root}/admin/models/unload"))
        .send()
        .await
        .map_err(|error| error.to_string())?;
    let status = response.status();
    let body = response
        .json::<Value>()
        .await
        .map_err(|error| error.to_string())?;
    if !status.is_success() {
        return Err(body
            .get("detail")
            .and_then(Value::as_str)
            .unwrap_or("Model unload failed")
            .to_string());
    }
    Ok(body)
}

#[tauri::command]
pub async fn engine_status(state: State<'_, AppState>) -> Result<Value, String> {
    let base_url = state
        .runtime
        .lock()
        .map_err(|_| "Runtime state unavailable".to_string())?
        .base_url();
    let Some(base_url) = base_url else {
        return Ok(json!({
            "ok": true,
            "running": false,
            "loaded": false,
            "model": null,
            "device": "mps",
            "tools": [],
            "baseUrl": null,
        }));
    };
    let response = state
        .client
        .get(format!("{}/health", base_url.trim_end_matches("/v1")))
        .timeout(Duration::from_secs(4))
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(format!("Engine returned {}", response.status()));
    }
    let mut status = response
        .json::<Value>()
        .await
        .map_err(|error| error.to_string())?;
    if let Some(object) = status.as_object_mut() {
        object.insert("running".to_string(), Value::Bool(true));
        object.insert("baseUrl".to_string(), Value::String(base_url));
    }
    Ok(status)
}

pub(crate) fn server_base_url(state: &AppState) -> Result<String, String> {
    state
        .runtime
        .lock()
        .map_err(|_| "Runtime state unavailable".to_string())?
        .base_url()
        .ok_or_else(|| "The local MLX engine is not running. Load a model first.".to_string())
}
