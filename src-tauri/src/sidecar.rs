use std::path::Path;

use tauri::AppHandle;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;
use tokio::sync::mpsc::Receiver;

pub struct SidecarHandle {
    pub rx: Receiver<CommandEvent>,
    pub child: CommandChild,
}

/// UNVERIFIED ON A PACKAGED BUILD — macOS TCC and the ffmpeg child process.
///
/// Since local files began going through ffmpeg, this is the first time a CHILD
/// PROCESS opens a path OUTSIDE the app's own data directory: the user's file,
/// wherever they picked or dropped it from. macOS gates `~/Documents`,
/// `~/Desktop`, `~/Downloads` and external volumes per *responsible process*, and
/// the implicit grant the user gives by choosing a file in an open panel is not
/// reliably inherited by a process the app spawns. If it is not, ffmpeg gets
/// EPERM on the input and the job fails with a permissions error that says
/// nothing useful.
///
/// `npm run dev` MASKS this completely: the responsible process is then the
/// terminal, which usually already holds those grants. Only a signed and
/// notarized build — the real responsible process, with the real entitlements —
/// settles it.
///
/// To verify: package the app, then transcribe a file from `~/Documents`, from
/// `~/Desktop` and from an external volume. If it fails, the fix is on this side
/// of the boundary — Rust reads the file itself (it already holds the grant, and
/// already streams it to hash it) and hands ffmpeg the bytes on stdin, rather
/// than a path the child has to open for itself.
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
