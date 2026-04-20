//! End-to-end tests for `CargoVerifier` — spawn a real temp crate and drive
//! the verifier against it so we catch regressions in manifest discovery,
//! subprocess handling, output truncation, and scope selection.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use runtime::{CargoVerifier, CargoVerifierConfig, VerificationResult, Verifier};

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
    format!(r#"{{"file_path":"{}"}}"#, path.display())
}

fn check_only() -> CargoVerifierConfig {
    CargoVerifierConfig {
        run_check: true,
        run_clippy: false,
        run_fmt: false,
        run_test: false,
        timeout: Duration::from_mins(2),
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

#[test]
fn passing_crate_reports_ok_and_passed_true() {
    if !cargo_available() {
        eprintln!("cargo unavailable — skipping");
        return;
    }
    let root = unique_tmpdir("pass");
    write_minimal_crate(&root, "vpass", "pub fn two() -> i32 { 2 }\n");
    let file = root.join("src/lib.rs");

    let v = CargoVerifier::new(check_only());
    let result: VerificationResult = v
        .verify("edit_file", &tool_input(&file))
        .expect("verifier should run for .rs edit");

    assert!(result.passed, "summary was: {}", result.summary);
    assert!(result.summary.contains("cargo check: ok"));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn type_error_fails_and_surfaces_error_text_in_summary() {
    if !cargo_available() {
        return;
    }
    let root = unique_tmpdir("typeerr");
    // Type mismatch: declared i32 but returns &str.
    write_minimal_crate(&root, "vtype", "pub fn oops() -> i32 { \"nope\" }\n");
    let file = root.join("src/lib.rs");

    let v = CargoVerifier::new(check_only());
    let result = v.verify("write_file", &tool_input(&file)).unwrap();

    assert!(!result.passed);
    assert!(result.summary.contains("cargo check: FAIL"));
    let lower = result.summary.to_lowercase();
    assert!(
        lower.contains("mismatched") || lower.contains("error"),
        "summary missing diagnostic: {}",
        result.summary
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn non_rust_file_is_out_of_scope() {
    let v = CargoVerifier::new(CargoVerifierConfig::default());
    let input = r#"{"file_path":"/tmp/README.md"}"#;
    assert!(v.verify("edit_file", input).is_none());
}

#[test]
fn unknown_tool_is_ignored() {
    let v = CargoVerifier::new(CargoVerifierConfig::default());
    let input = r#"{"file_path":"/tmp/x.rs"}"#;
    assert!(v.verify("read_file", input).is_none());
    assert!(v.verify("bash", input).is_none());
}

#[test]
fn malformed_json_returns_none_without_panicking() {
    let v = CargoVerifier::new(CargoVerifierConfig::default());
    assert!(v.verify("edit_file", "not-json").is_none());
    assert!(v.verify("edit_file", "{}").is_none());
    assert!(v.verify("edit_file", r#"{"file_path": 42}"#).is_none());
}

#[test]
fn accepts_alternate_path_keys() {
    if !cargo_available() {
        return;
    }
    let root = unique_tmpdir("altkeys");
    write_minimal_crate(&root, "valt", "pub fn k() -> u8 { 1 }\n");
    let file = root.join("src/lib.rs");

    let v = CargoVerifier::new(check_only());

    let a = format!(r#"{{"filePath":"{}"}}"#, file.display());
    let b = format!(r#"{{"path":"{}"}}"#, file.display());
    assert!(v.verify("edit_file", &a).is_some());
    assert!(v.verify("edit_file", &b).is_some());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn file_outside_any_crate_is_skipped() {
    let root = unique_tmpdir("nocargo");
    // No Cargo.toml anywhere up the tree we control — but the tmp root's
    // ancestors might have one. To guarantee "none found", create a file
    // whose parent chain hits filesystem root without Cargo.toml only when
    // the OS tmpdir itself isn't under a cargo project. Skip the strong
    // assertion in that case.
    fs::create_dir_all(root.join("x")).unwrap();
    let file = root.join("x/orphan.rs");
    fs::write(&file, "pub fn z() {}\n").unwrap();

    let v = CargoVerifier::new(check_only());
    let result = v.verify("edit_file", &tool_input(&file));
    // Either no manifest found (None) — the preferred case — or one was
    // discovered in an ancestor; both are acceptable. We just assert it
    // doesn't panic and returns a well-formed value.
    if let Some(r) = result {
        assert!(r.summary.contains("cargo check"));
    }
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn all_steps_disabled_yields_none() {
    let config = CargoVerifierConfig {
        run_check: false,
        run_clippy: false,
        run_fmt: false,
        run_test: false,
        timeout: Duration::from_secs(5),
    };
    let root = unique_tmpdir("nosteps");
    write_minimal_crate(&root, "vnone", "pub fn n() {}\n");
    let file = root.join("src/lib.rs");

    let v = CargoVerifier::new(config);
    assert!(v.verify("edit_file", &tool_input(&file)).is_none());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn timeout_short_circuits_and_reports_failure() {
    if !cargo_available() {
        return;
    }
    // 1ms timeout — cargo process spawn alone takes longer than this on any
    // real machine, so the verifier should report a timeout failure.
    let config = CargoVerifierConfig {
        run_check: true,
        run_clippy: false,
        run_fmt: false,
        run_test: false,
        timeout: Duration::from_millis(1),
    };
    let root = unique_tmpdir("timeout");
    write_minimal_crate(&root, "vtime", "pub fn t() {}\n");
    let file = root.join("src/lib.rs");

    let v = CargoVerifier::new(config);
    let result = v.verify("edit_file", &tool_input(&file)).unwrap();
    assert!(!result.passed);
    assert!(
        result.summary.contains("cargo check: FAIL")
            && result.summary.to_lowercase().contains("timed out"),
        "unexpected summary: {}",
        result.summary
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn later_steps_are_skipped_after_first_failure() {
    if !cargo_available() {
        return;
    }
    let root = unique_tmpdir("skipchain");
    // Broken code so `cargo check` fails — the later steps (clippy/fmt/test)
    // must all be recorded as `skipped` to save time.
    write_minimal_crate(&root, "vskip", "pub fn bad() -> i32 { return; }\n");
    let file = root.join("src/lib.rs");

    let config = CargoVerifierConfig {
        run_check: true,
        run_clippy: true,
        run_fmt: true,
        run_test: true,
        timeout: Duration::from_mins(2),
    };
    let v = CargoVerifier::new(config);
    let result = v.verify("edit_file", &tool_input(&file)).unwrap();

    assert!(!result.passed);
    assert!(result.summary.contains("cargo check: FAIL"));
    assert!(result.summary.contains("cargo clippy: skipped"));
    assert!(result.summary.contains("cargo fmt --check: skipped"));
    assert!(result.summary.contains("cargo test: skipped"));
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn fmt_violation_is_detected_when_fmt_enabled() {
    if !cargo_available() {
        return;
    }
    let root = unique_tmpdir("fmt");
    // Deliberately badly-formatted source that still compiles.
    let src = "pub    fn   f( )->i32{1   +  2}\n";
    write_minimal_crate(&root, "vfmt", src);
    let file = root.join("src/lib.rs");

    let config = CargoVerifierConfig {
        run_check: false,
        run_clippy: false,
        run_fmt: true,
        run_test: false,
        timeout: Duration::from_mins(1),
    };
    let v = CargoVerifier::new(config);
    let result = v.verify("edit_file", &tool_input(&file)).unwrap();

    // rustfmt may be unavailable in some toolchains — accept both "FAIL"
    // (violations found) and "could not run" (binary missing), but never ok.
    assert!(!result.passed, "summary: {}", result.summary);
    assert!(result.summary.contains("cargo fmt --check"));
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
    fs::write(root.join("src/lib.rs"), "pub mod sub;\n").unwrap();
    fs::remove_file(root.join("src/lib.rs")).ok();
    fs::write(
        root.join("src/lib.rs"),
        "#[path = \"sub/mod.rs\"]\npub mod sub;\n",
    )
    .unwrap();

    let v = CargoVerifier::new(check_only());
    let result = v.verify("edit_file", &tool_input(&nested)).unwrap();
    assert!(result.passed, "summary: {}", result.summary);
    let _ = fs::remove_dir_all(&root);
}
