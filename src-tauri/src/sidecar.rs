use std::path::Path;

use tauri::AppHandle;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;
use tokio::sync::mpsc::Receiver;

pub struct SidecarHandle {
    pub rx: Receiver<CommandEvent>,
    pub child: CommandChild,
}

pub fn spawn_sidecar(
    app: &AppHandle,
    name: &str,
    args: &[&str],
    cwd: Option<&Path>,
) -> anyhow::Result<SidecarHandle> {
    let mut command = app.shell().sidecar(name)?;
    command = command.args(args);
    if let Some(dir) = cwd {
        command = command.current_dir(dir);
    }
    let (rx, child) = command.spawn()?;
    Ok(SidecarHandle { rx, child })
}
