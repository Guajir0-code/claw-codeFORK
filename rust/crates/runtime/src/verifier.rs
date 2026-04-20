//! Self-verification of code edits.
//!
//! After the model runs a writing tool (`edit_file` / `write_file`) on a Rust
//! source file, a [`Verifier`] is given the chance to run additional checks
//! (`cargo check`, `cargo clippy`, `cargo fmt --check`, `cargo test`) against
//! the affected crate and feed the result back into the tool output. The
//! assistant then sees any compilation, lint, formatting, or test failures on
//! the very next iteration and can correct them without the user having to
//! intervene.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde_json::Value;

/// Output of a single verification run injected back into the tool result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationResult {
    pub passed: bool,
    pub summary: String,
}

/// Strategy that inspects a completed tool invocation and optionally produces
/// additional diagnostics to inject into the tool result.
pub trait Verifier: Send {
    /// Return `Some` when the tool/input pair is in scope for verification,
    /// `None` otherwise (e.g. a `read_file` call, or an edit on `README.md`).
    fn verify(&self, tool_name: &str, tool_input: &str) -> Option<VerificationResult>;
}

/// Declarative configuration for the built-in cargo-based verifier.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CargoVerifierConfig {
    pub run_check: bool,
    pub run_clippy: bool,
    pub run_fmt: bool,
    pub run_test: bool,
    pub timeout: Duration,
}

impl Default for CargoVerifierConfig {
    fn default() -> Self {
        Self {
            run_check: true,
            run_clippy: true,
            run_fmt: true,
            run_test: true,
            timeout: Duration::from_mins(2),
        }
    }
}

/// Maximum bytes kept per cargo step before the summary is truncated.
const MAX_STEP_OUTPUT_BYTES: usize = 2_048;

/// Tool names recognised as writing to a file on disk.
const WRITE_TOOLS: &[&str] = &["edit_file", "write_file", "Edit", "Write"];

/// Built-in verifier that drives cargo against the crate that owns the
/// edited file. Runs checks sequentially with early-exit on the first failure.
pub struct CargoVerifier {
    config: CargoVerifierConfig,
}

impl CargoVerifier {
    #[must_use]
    pub fn new(config: CargoVerifierConfig) -> Self {
        Self { config }
    }
}

impl Verifier for CargoVerifier {
    fn verify(&self, tool_name: &str, tool_input: &str) -> Option<VerificationResult> {
        if !WRITE_TOOLS.contains(&tool_name) {
            return None;
        }
        let file_path = extract_file_path(tool_input)?;
        if file_path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            return None;
        }
        let manifest = nearest_cargo_manifest(&file_path)?;

        let steps = planned_steps(&self.config);
        if steps.is_empty() {
            return None;
        }

        let mut summary = String::new();
        let mut overall_passed = true;
        let mut skip_remaining = false;

        for step in steps {
            if skip_remaining {
                writeln!(summary, "[verifier] {}: skipped", step.label).ok();
                continue;
            }
            let outcome = run_step(&step, &manifest, self.config.timeout);
            match outcome {
                StepOutcome::Passed => {
                    writeln!(summary, "[verifier] {}: ok", step.label).ok();
                }
                StepOutcome::Failed { body } => {
                    overall_passed = false;
                    skip_remaining = true;
                    writeln!(summary, "[verifier] {}: FAIL", step.label).ok();
                    summary.push_str(&truncate_output(&body));
                    summary.push('\n');
                }
                StepOutcome::Unavailable { message } => {
                    overall_passed = false;
                    skip_remaining = true;
                    writeln!(
                        summary,
                        "[verifier] {}: could not run ({message})",
                        step.label
                    )
                    .ok();
                }
            }
        }

        Some(VerificationResult {
            passed: overall_passed,
            summary: summary.trim_end().to_string(),
        })
    }
}

/// Prepend the verifier summary to the tool output using a visible delimiter.
#[must_use]
pub fn prepend_verifier_summary(summary: &str, output: String) -> String {
    if summary.is_empty() {
        return output;
    }
    if output.trim().is_empty() {
        return summary.to_string();
    }
    format!("{output}\n\n[verifier output]\n{summary}")
}

fn extract_file_path(input: &str) -> Option<PathBuf> {
    let value: Value = serde_json::from_str(input).ok()?;
    let path_str = value
        .get("file_path")
        .or_else(|| value.get("filePath"))
        .or_else(|| value.get("path"))?
        .as_str()?;
    Some(PathBuf::from(path_str))
}

fn nearest_cargo_manifest(file_path: &Path) -> Option<PathBuf> {
    let start = if file_path.is_absolute() {
        file_path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(file_path)
    };
    let mut cursor = start.parent()?.to_path_buf();
    loop {
        let candidate = cursor.join("Cargo.toml");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !cursor.pop() {
            return None;
        }
    }
}

#[derive(Debug, Clone)]
struct PlannedStep {
    label: &'static str,
    args: Vec<&'static str>,
}

enum StepOutcome {
    Passed,
    Failed { body: String },
    Unavailable { message: String },
}

fn planned_steps(config: &CargoVerifierConfig) -> Vec<PlannedStep> {
    let mut steps = Vec::new();
    if config.run_check {
        steps.push(PlannedStep {
            label: "cargo check",
            args: vec!["check", "--quiet", "--message-format=short"],
        });
    }
    if config.run_clippy {
        steps.push(PlannedStep {
            label: "cargo clippy",
            args: vec![
                "clippy",
                "--quiet",
                "--message-format=short",
                "--",
                "-D",
                "warnings",
            ],
        });
    }
    if config.run_fmt {
        steps.push(PlannedStep {
            label: "cargo fmt --check",
            args: vec!["fmt", "--", "--check"],
        });
    }
    if config.run_test {
        steps.push(PlannedStep {
            label: "cargo test",
            args: vec!["test", "--quiet", "--no-fail-fast"],
        });
    }
    steps
}

fn run_step(step: &PlannedStep, manifest: &Path, timeout: Duration) -> StepOutcome {
    let mut command = Command::new("cargo");
    command.arg(step.args[0]);
    command.arg("--manifest-path").arg(manifest);
    for arg in step.args.iter().skip(1) {
        command.arg(arg);
    }
    command.env("CARGO_TERM_COLOR", "never");

    match spawn_with_timeout(command, timeout) {
        Ok(output) => {
            if output.status.success() {
                StepOutcome::Passed
            } else {
                let mut body = String::new();
                body.push_str(&String::from_utf8_lossy(&output.stdout));
                if !output.stderr.is_empty() {
                    if !body.is_empty() {
                        body.push('\n');
                    }
                    body.push_str(&String::from_utf8_lossy(&output.stderr));
                }
                if body.trim().is_empty() {
                    body = format!("exit status: {}", output.status);
                }
                StepOutcome::Failed { body }
            }
        }
        Err(SpawnError::Timeout) => StepOutcome::Failed {
            body: format!("step timed out after {}s", timeout.as_secs()),
        },
        Err(SpawnError::Io(error)) => StepOutcome::Unavailable {
            message: error.to_string(),
        },
    }
}

enum SpawnError {
    Timeout,
    Io(std::io::Error),
}

fn spawn_with_timeout(
    mut command: Command,
    timeout: Duration,
) -> Result<std::process::Output, SpawnError> {
    use std::sync::mpsc;
    use std::thread;

    command.stdin(std::process::Stdio::null());
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::piped());
    let mut child = command.spawn().map_err(SpawnError::Io)?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let (tx, rx) = mpsc::channel();
    let stdout_handle = stdout.map(|mut s| {
        let tx = tx.clone();
        thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = std::io::Read::read_to_end(&mut s, &mut buf);
            let _ = tx.send(("stdout", buf));
        })
    });
    let stderr_handle = stderr.map(|mut s| {
        let tx = tx.clone();
        thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = std::io::Read::read_to_end(&mut s, &mut buf);
            let _ = tx.send(("stderr", buf));
        })
    });
    drop(tx);

    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().map_err(SpawnError::Io)? {
            if let Some(h) = stdout_handle {
                let _ = h.join();
            }
            if let Some(h) = stderr_handle {
                let _ = h.join();
            }
            let mut stdout_bytes = Vec::new();
            let mut stderr_bytes = Vec::new();
            while let Ok((which, bytes)) = rx.try_recv() {
                if which == "stdout" {
                    stdout_bytes = bytes;
                } else {
                    stderr_bytes = bytes;
                }
            }
            return Ok(std::process::Output {
                status,
                stdout: stdout_bytes,
                stderr: stderr_bytes,
            });
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(SpawnError::Timeout);
        }
        thread::sleep(Duration::from_millis(50));
    }
}

/// Trim output to `MAX_STEP_OUTPUT_BYTES`, preserving the head plus any lines
/// containing `error` or `warning` so the model keeps the actionable signal.
fn truncate_output(body: &str) -> String {
    if body.len() <= MAX_STEP_OUTPUT_BYTES {
        return body.to_string();
    }
    let head_budget = MAX_STEP_OUTPUT_BYTES * 2 / 3;
    let mut head = String::new();
    for line in body.lines() {
        if head.len() + line.len() + 1 > head_budget {
            break;
        }
        head.push_str(line);
        head.push('\n');
    }
    let mut signal_lines = Vec::new();
    for line in body.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("error") || lower.contains("warning") {
            signal_lines.push(line);
        }
    }
    let mut tail = String::new();
    let tail_budget = MAX_STEP_OUTPUT_BYTES - head.len();
    for line in signal_lines.into_iter().rev() {
        if tail.len() + line.len() + 1 > tail_budget {
            break;
        }
        tail = format!("{line}\n{tail}");
    }
    format!("{head}... (truncated) ...\n{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn verify_returns_none_for_non_write_tools() {
        let v = CargoVerifier::new(CargoVerifierConfig::default());
        assert!(v.verify("read_file", r#"{"file_path":"lib.rs"}"#).is_none());
        assert!(v.verify("grep_search", "{}").is_none());
    }

    #[test]
    fn verify_returns_none_for_non_rust_extension() {
        let v = CargoVerifier::new(CargoVerifierConfig::default());
        assert!(v
            .verify("edit_file", r#"{"file_path":"notes.md"}"#)
            .is_none());
        assert!(v
            .verify("write_file", r#"{"file_path":"config.json"}"#)
            .is_none());
    }

    #[test]
    fn verify_returns_none_when_no_manifest_found() {
        let v = CargoVerifier::new(CargoVerifierConfig {
            run_check: true,
            run_clippy: false,
            run_fmt: false,
            run_test: false,
            timeout: Duration::from_secs(1),
        });
        // /tmp has no Cargo.toml up the tree.
        let input = r#"{"file_path":"/tmp/__claw_verifier_test_missing.rs"}"#;
        assert!(v.verify("edit_file", input).is_none());
    }

    #[test]
    fn extract_file_path_supports_multiple_key_names() {
        assert_eq!(
            extract_file_path(r#"{"file_path":"a.rs"}"#),
            Some(PathBuf::from("a.rs"))
        );
        assert_eq!(
            extract_file_path(r#"{"filePath":"b.rs"}"#),
            Some(PathBuf::from("b.rs"))
        );
        assert_eq!(
            extract_file_path(r#"{"path":"c.rs"}"#),
            Some(PathBuf::from("c.rs"))
        );
        assert_eq!(extract_file_path("not json"), None);
        assert_eq!(extract_file_path(r#"{"other":1}"#), None);
    }

    #[test]
    fn nearest_cargo_manifest_walks_up_directories() {
        let tmp = tempdir();
        let crate_root = tmp.join("my_crate");
        let nested = crate_root.join("src").join("nested");
        fs::create_dir_all(&nested).unwrap();
        fs::write(crate_root.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
        let file = nested.join("thing.rs");
        fs::write(&file, "").unwrap();

        let manifest = nearest_cargo_manifest(&file).expect("manifest should be found");
        assert_eq!(manifest, crate_root.join("Cargo.toml"));
    }

    #[test]
    fn truncate_preserves_error_lines() {
        let mut body = String::new();
        for i in 0..2_000 {
            writeln!(body, "noise line {i}").unwrap();
        }
        body.push_str("error[E0308]: mismatched types\n");
        let truncated = truncate_output(&body);
        assert!(truncated.len() <= MAX_STEP_OUTPUT_BYTES + 64);
        assert!(truncated.contains("error[E0308]"));
        assert!(truncated.contains("... (truncated) ..."));
    }

    #[test]
    fn prepend_verifier_summary_merges_with_existing_output() {
        let merged = prepend_verifier_summary("[verifier] cargo check: ok", "edited 1 file".into());
        assert!(merged.contains("edited 1 file"));
        assert!(merged.contains("[verifier output]"));
        assert!(merged.contains("cargo check: ok"));
    }

    #[test]
    fn prepend_verifier_summary_passes_through_empty_summary() {
        let merged = prepend_verifier_summary("", "edited".into());
        assert_eq!(merged, "edited");
    }

    fn tempdir() -> PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "claw_verifier_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = base.join(unique);
        fs::create_dir_all(&path).unwrap();
        path
    }
}
