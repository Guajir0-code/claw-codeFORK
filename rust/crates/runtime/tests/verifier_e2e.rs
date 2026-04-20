//! End-to-end tests for `CargoVerifier` using real temp projects.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use runtime::{
    CargoVerifier, CargoVerifierConfig, VerificationContext, VerificationFailureKind,
    VerificationPhase, VerificationReport, VerificationStatus, Verifier,
};
use serde_json::json;

static TMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn unique_tmpdir(tag: &str) -> PathBuf {
    let pid = std::process::id();
    let n = TMP_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("verifier_e2e_{tag}_{pid}_{n}"));
    if dir.exists() {
        let _ = fs::remove_dir_all(&dir);
    }
    fs::create_dir_all(&dir).expect("tmpdir");
    dir
}

fn write_minimal_crate(root: &Path, name: &str, lib_rs: &str) {
    fs::write(
        root.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n"
        ),
    )
    .unwrap();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/lib.rs"), lib_rs).unwrap();
}

fn tool_input(path: &Path) -> String {
    json!({ "file_path": path }).to_string()
}

fn context_for(tool_name: &str, input: &str) -> Option<VerificationContext> {
    VerificationContext::from_tool_invocation(VerificationPhase::Quick, None, tool_name, input, 1)
}

fn quick_report(v: &CargoVerifier, tool_name: &str, input: &str) -> Option<VerificationReport> {
    let context = context_for(tool_name, input)?;
    v.quick_verify(&context).into_iter().next()
}

fn quick_only() -> CargoVerifierConfig {
    CargoVerifierConfig {
        legacy_mode: false,
        quick_on_write: true,
        final_gate: false,
        max_output_bytes: 2_048,
        rust_check: true,
        rust_clippy: false,
        rust_fmt: false,
        rust_test: false,
        rust_timeout: Duration::from_mins(2),
        node_enabled: false,
        node_timeout: Duration::from_mins(2),
        python_enabled: false,
        python_timeout: Duration::from_mins(2),
    }
}

fn cargo_available() -> bool {
    std::process::Command::new("cargo")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn python_available() -> bool {
    std::process::Command::new("python")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn python_only() -> CargoVerifierConfig {
    CargoVerifierConfig {
        legacy_mode: false,
        quick_on_write: true,
        final_gate: true,
        max_output_bytes: 2_048,
        rust_check: false,
        rust_clippy: false,
        rust_fmt: false,
        rust_test: false,
        rust_timeout: Duration::from_mins(2),
        node_enabled: false,
        node_timeout: Duration::from_mins(2),
        python_enabled: true,
        python_timeout: Duration::from_mins(2),
    }
}

#[test]
fn passing_crate_reports_passed_status() {
    if !cargo_available() {
        eprintln!("cargo unavailable - skipping");
        return;
    }
    let root = unique_tmpdir("pass");
    write_minimal_crate(&root, "vpass", "pub fn two() -> i32 { 2 }\n");
    let file = root.join("src/lib.rs");

    let v = CargoVerifier::new(quick_only());
    let result = quick_report(&v, "edit_file", &tool_input(&file))
        .expect("verifier should run for .rs edit");

    assert_eq!(result.status, VerificationStatus::Passed);
    assert!(result.summary_text.contains("cargo check: ok"));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn type_error_fails_and_surfaces_error_text_in_summary() {
    if !cargo_available() {
        return;
    }
    let root = unique_tmpdir("typeerr");
    write_minimal_crate(&root, "vtype", "pub fn oops() -> i32 { \"nope\" }\n");
    let file = root.join("src/lib.rs");

    let v = CargoVerifier::new(quick_only());
    let result = quick_report(&v, "write_file", &tool_input(&file)).unwrap();

    assert_eq!(result.status, VerificationStatus::Failed);
    assert!(result.summary_text.contains("cargo check: FAIL"));
    let lower = result.summary_text.to_lowercase();
    assert!(
        lower.contains("mismatched") || lower.contains("error"),
        "summary missing diagnostic: {}",
        result.summary_text
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn non_rust_file_is_out_of_scope() {
    let v = CargoVerifier::new(CargoVerifierConfig::default());
    let input = r#"{"file_path":"/tmp/README.md"}"#;
    assert!(quick_report(&v, "edit_file", input).is_none());
}

#[test]
fn unknown_tool_is_ignored() {
    let v = CargoVerifier::new(CargoVerifierConfig::default());
    let input = r#"{"file_path":"/tmp/x.rs"}"#;
    assert!(quick_report(&v, "read_file", input).is_none());
    assert!(quick_report(&v, "bash", input).is_none());
}

#[test]
fn malformed_json_returns_none_without_panicking() {
    let v = CargoVerifier::new(CargoVerifierConfig::default());
    assert!(quick_report(&v, "edit_file", "not-json").is_none());
    assert!(quick_report(&v, "edit_file", "{}").is_none());
    assert!(quick_report(&v, "edit_file", r#"{"file_path": 42}"#).is_none());
}

#[test]
fn accepts_alternate_path_keys() {
    if !cargo_available() {
        return;
    }
    let root = unique_tmpdir("altkeys");
    write_minimal_crate(&root, "valt", "pub fn k() -> u8 { 1 }\n");
    let file = root.join("src/lib.rs");

    let v = CargoVerifier::new(quick_only());

    let a = json!({ "filePath": file }).to_string();
    let b = json!({ "path": file }).to_string();
    assert!(quick_report(&v, "edit_file", &a).is_some());
    assert!(quick_report(&v, "edit_file", &b).is_some());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn file_outside_any_crate_is_skipped_or_ignored() {
    let root = unique_tmpdir("nocargo");
    fs::create_dir_all(root.join("x")).unwrap();
    let file = root.join("x/orphan.rs");
    fs::write(&file, "pub fn z() {}\n").unwrap();

    let v = CargoVerifier::new(quick_only());
    let result = quick_report(&v, "edit_file", &tool_input(&file));
    if let Some(report) = result {
        assert!(report.summary_text.contains("cargo check"));
    }
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn all_steps_disabled_yields_skipped_report() {
    let config = CargoVerifierConfig {
        rust_check: false,
        rust_clippy: false,
        rust_fmt: false,
        rust_test: false,
        ..quick_only()
    };
    let root = unique_tmpdir("nosteps");
    write_minimal_crate(&root, "vnone", "pub fn n() {}\n");
    let file = root.join("src/lib.rs");

    let v = CargoVerifier::new(config);
    let report = quick_report(&v, "edit_file", &tool_input(&file)).unwrap();
    assert_eq!(report.status, VerificationStatus::Skipped);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn timeout_short_circuits_and_reports_failure() {
    if !cargo_available() {
        return;
    }
    let config = CargoVerifierConfig {
        rust_timeout: Duration::from_millis(1),
        ..quick_only()
    };
    let root = unique_tmpdir("timeout");
    write_minimal_crate(&root, "vtime", "pub fn t() {}\n");
    let file = root.join("src/lib.rs");

    let v = CargoVerifier::new(config);
    let result = quick_report(&v, "edit_file", &tool_input(&file)).unwrap();
    assert_eq!(result.status, VerificationStatus::Failed);
    assert!(
        result.summary_text.contains("cargo check: FAIL")
            && result.summary_text.to_lowercase().contains("timed out"),
        "unexpected summary: {}",
        result.summary_text
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn later_steps_are_skipped_after_first_failure_in_legacy_mode() {
    if !cargo_available() {
        return;
    }
    let root = unique_tmpdir("skipchain");
    write_minimal_crate(&root, "vskip", "pub fn bad() -> i32 { return; }\n");
    let file = root.join("src/lib.rs");

    let config = CargoVerifierConfig {
        legacy_mode: true,
        quick_on_write: true,
        final_gate: false,
        max_output_bytes: 2_048,
        rust_check: true,
        rust_clippy: true,
        rust_fmt: true,
        rust_test: true,
        rust_timeout: Duration::from_mins(2),
        node_enabled: false,
        node_timeout: Duration::from_mins(2),
        python_enabled: false,
        python_timeout: Duration::from_mins(2),
    };
    let v = CargoVerifier::new(config);
    let result = quick_report(&v, "edit_file", &tool_input(&file)).unwrap();

    assert_eq!(result.status, VerificationStatus::Failed);
    assert!(result.summary_text.contains("cargo check: FAIL"));
    assert!(result.summary_text.contains("cargo clippy: skipped"));
    assert!(result.summary_text.contains("cargo fmt --check: skipped"));
    assert!(result.summary_text.contains("cargo test: skipped"));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn fmt_violation_is_detected_when_fmt_enabled() {
    if !cargo_available() {
        return;
    }
    let root = unique_tmpdir("fmt");
    let src = "pub    fn   f( )->i32{1   +  2}\n";
    write_minimal_crate(&root, "vfmt", src);
    let file = root.join("src/lib.rs");

    let config = CargoVerifierConfig {
        legacy_mode: true,
        quick_on_write: true,
        final_gate: false,
        max_output_bytes: 2_048,
        rust_check: false,
        rust_clippy: false,
        rust_fmt: true,
        rust_test: false,
        rust_timeout: Duration::from_mins(1),
        node_enabled: false,
        node_timeout: Duration::from_mins(2),
        python_enabled: false,
        python_timeout: Duration::from_mins(2),
    };
    let v = CargoVerifier::new(config);
    let result = quick_report(&v, "edit_file", &tool_input(&file)).unwrap();

    assert_ne!(result.status, VerificationStatus::Passed);
    assert!(result.summary_text.contains("cargo fmt --check"));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn nested_file_resolves_to_parent_crate_manifest() {
    if !cargo_available() {
        return;
    }
    let root = unique_tmpdir("nested");
    write_minimal_crate(&root, "vnest", "pub mod sub;\n");
    fs::create_dir_all(root.join("src/sub")).unwrap();
    let nested = root.join("src/sub/mod.rs");
    fs::write(&nested, "pub fn inside() -> u8 { 7 }\n").unwrap();
    fs::write(
        root.join("src/lib.rs"),
        "#[path = \"sub/mod.rs\"]\npub mod sub;\n",
    )
    .unwrap();

    let v = CargoVerifier::new(quick_only());
    let result = quick_report(&v, "edit_file", &tool_input(&nested)).unwrap();
    assert_eq!(result.status, VerificationStatus::Passed);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn python_quick_fallback_py_compile_catches_syntax_error() {
    if !python_available() {
        return;
    }
    let root = unique_tmpdir("python-syntax");
    fs::write(root.join("requirements.txt"), "pytest\n").unwrap();
    let file = root.join("broken.py");
    fs::write(&file, "def broken(:\n    pass\n").unwrap();

    let v = CargoVerifier::new(python_only());
    let result = quick_report(&v, "edit_file", &tool_input(&file)).unwrap();

    assert_eq!(result.status, VerificationStatus::Failed);
    assert_eq!(
        result.steps[0].failure_kind,
        Some(VerificationFailureKind::Code)
    );
    assert!(result.summary_text.contains("py_compile"));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn invalid_pyproject_reports_config_failure() {
    let root = unique_tmpdir("python-bad-pyproject");
    let file = root.join("pyproject.toml");
    fs::write(&file, "[tool.ruff\n").unwrap();

    let v = CargoVerifier::new(python_only());
    let result = quick_report(&v, "edit_file", &tool_input(&file)).unwrap();

    assert_eq!(result.status, VerificationStatus::Failed);
    assert_eq!(
        result.steps[0].failure_kind,
        Some(VerificationFailureKind::Config)
    );
    assert!(result.summary_text.contains("pyproject.toml"));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn broken_local_venv_is_reported_as_environment() {
    let root = unique_tmpdir("python-venv");
    fs::write(root.join("setup.py"), "from setuptools import setup\n").unwrap();
    let file = root.join("main.py");
    fs::write(&file, "print('ok')\n").unwrap();

    let interpreter = if cfg!(windows) {
        root.join(".venv").join("Scripts").join("python.exe")
    } else {
        root.join(".venv").join("bin").join("python")
    };
    fs::create_dir_all(interpreter.parent().unwrap()).unwrap();
    fs::write(&interpreter, "").unwrap();

    let v = CargoVerifier::new(python_only());
    let result = quick_report(&v, "edit_file", &tool_input(&file)).unwrap();

    assert!(
        matches!(
            result.steps[0].failure_kind,
            Some(VerificationFailureKind::Environment)
        ),
        "expected environment failure, got {:?}",
        result.steps[0].failure_kind
    );
    let _ = fs::remove_dir_all(&root);
}
