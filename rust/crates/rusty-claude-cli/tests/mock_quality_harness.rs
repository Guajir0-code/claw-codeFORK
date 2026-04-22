use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use mock_anthropic_service::{MockAnthropicService, SCENARIO_PREFIX};
use serde_json::{json, Value};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[cfg(unix)]
fn make_executable(path: &Path) {
    let mut permissions = fs::metadata(path).expect("script metadata").permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).expect("script should be executable");
}

#[cfg(not(unix))]
fn make_executable(path: &Path) {
    let _ = path;
}

fn program_name(stem: &str) -> String {
    if cfg!(windows) {
        format!("{stem}.cmd")
    } else {
        stem.to_string()
    }
}

fn write_program_stub(dir: &Path, stem: &str, unix_body: &str, windows_ps1: &str) {
    if cfg!(windows) {
        let ps1_path = dir.join(format!("{stem}.ps1"));
        fs::write(&ps1_path, windows_ps1).expect("powershell stub should write");
        let wrapper = format!(
            "@echo off\r\npowershell -NoProfile -ExecutionPolicy Bypass -File \"%~dp0\\{stem}.ps1\" %*\r\n"
        );
        let cmd_path = dir.join(program_name(stem));
        fs::write(&cmd_path, wrapper).expect("cmd wrapper should write");
    } else {
        let script_path = dir.join(program_name(stem));
        fs::write(&script_path, unix_body).expect("shell stub should write");
        make_executable(&script_path);
    }
}

fn configure_quality_process_env(command: &mut Command, workspace: &HarnessWorkspace) {
    if cfg!(windows) {
        command.env("USERPROFILE", &workspace.home);
        for key in [
            "SystemRoot",
            "ComSpec",
            "PATHEXT",
            "TEMP",
            "TMP",
            "CARGO_HOME",
            "RUSTUP_HOME",
            "VCINSTALLDIR",
            "VCToolsInstallDir",
            "VCToolsVersion",
            "VSINSTALLDIR",
            "VisualStudioVersion",
            "WindowsSdkDir",
            "WindowsSDKVersion",
            "UniversalCRTSdkDir",
            "UCRTVersion",
            "INCLUDE",
            "LIB",
            "LIBPATH",
        ] {
            if let Ok(value) = std::env::var(key) {
                command.env(key, value);
            }
        }
        let mut paths = vec![workspace.bin.display().to_string()];
        if let Ok(path) = std::env::var("PATH") {
            paths.push(path);
        }
        command.env("PATH", paths.join(";"));
    } else {
        let path = std::env::var("PATH").unwrap_or_else(|_| "/usr/bin:/bin".to_string());
        command.env("PATH", format!("{}:{path}", workspace.bin.display()));
        for key in ["CARGO_HOME", "RUSTUP_HOME"] {
            if let Ok(value) = std::env::var(key) {
                command.env(key, value);
            }
        }
    }
}

fn run_quality_case(case: ScenarioCase) {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime should build");
    let server = runtime
        .block_on(MockAnthropicService::spawn())
        .expect("mock service should start");
    let base_url = server.base_url();
    let workspace = HarnessWorkspace::new(unique_temp_dir(case.name));
    workspace.create().expect("workspace should exist");
    (case.prepare)(&workspace);

    let run = match run_case(case, &workspace, &base_url) {
        Ok(run) => run,
        Err(output) => {
            let captured = runtime.block_on(server.captured_requests());
            let messages_only = captured
                .iter()
                .filter(|request| request.path == "/v1/messages")
                .collect::<Vec<_>>();
            let last_request = messages_only
                .last()
                .map_or("<no /v1/messages request captured>", |request| {
                    request.raw_body.as_str()
                });
            panic!(
                "case {} failed\nmessages requests: {}\nstdout:\n{}\n\nstderr:\n{}\n\nlast request body:\n{}",
                case.name,
                messages_only.len(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
                last_request
            );
        }
    };
    (case.assert)(&workspace, &run);

    fs::remove_dir_all(&workspace.root).expect("workspace cleanup should succeed");
}

#[test]
fn quality_harness_rust_red_green() {
    run_quality_case(ScenarioCase {
        name: "rust_red_green",
        prepare: prepare_rust_red_green_fixture,
        assert: assert_rust_red_green,
    });
}

#[test]
fn quality_harness_node_red_green() {
    run_quality_case(ScenarioCase {
        name: "node_red_green",
        prepare: prepare_node_red_green_fixture,
        assert: assert_node_red_green,
    });
}

#[test]
fn quality_harness_python_red_green() {
    run_quality_case(ScenarioCase {
        name: "python_red_green",
        prepare: prepare_python_red_green_fixture,
        assert: assert_python_red_green,
    });
}

#[test]
fn quality_harness_rust_config_failure() {
    run_quality_case(ScenarioCase {
        name: "rust_config_failure",
        prepare: prepare_rust_config_failure_fixture,
        assert: assert_rust_config_failure,
    });
}

#[test]
fn quality_harness_node_tool_unavailable() {
    run_quality_case(ScenarioCase {
        name: "node_tool_unavailable",
        prepare: prepare_node_tool_unavailable_fixture,
        assert: assert_node_tool_unavailable,
    });
}

#[test]
fn quality_harness_python_timeout() {
    run_quality_case(ScenarioCase {
        name: "python_timeout",
        prepare: prepare_python_timeout_fixture,
        assert: assert_python_timeout,
    });
}

#[test]
fn quality_harness_rust_final_gate_retry() {
    run_quality_case(ScenarioCase {
        name: "rust_final_gate_retry",
        prepare: prepare_rust_final_gate_retry_fixture,
        assert: assert_rust_final_gate_retry,
    });
}

#[derive(Clone, Copy)]
struct ScenarioCase {
    name: &'static str,
    prepare: fn(&HarnessWorkspace),
    assert: fn(&HarnessWorkspace, &ScenarioRun),
}

struct HarnessWorkspace {
    root: PathBuf,
    config_home: PathBuf,
    home: PathBuf,
    bin: PathBuf,
}

impl HarnessWorkspace {
    fn new(root: PathBuf) -> Self {
        Self {
            config_home: root.join("config-home"),
            home: root.join("home"),
            bin: root.join("bin"),
            root,
        }
    }

    fn create(&self) -> std::io::Result<()> {
        fs::create_dir_all(&self.root)?;
        fs::create_dir_all(&self.config_home)?;
        fs::create_dir_all(&self.home)?;
        fs::create_dir_all(&self.bin)?;
        Ok(())
    }
}

struct ScenarioRun {
    response: Value,
}

fn run_case(
    case: ScenarioCase,
    workspace: &HarnessWorkspace,
    base_url: &str,
) -> Result<ScenarioRun, Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_claw"));
    command
        .current_dir(&workspace.root)
        .env_clear()
        .env("ANTHROPIC_API_KEY", "test-quality-key")
        .env("ANTHROPIC_BASE_URL", base_url)
        .env("CLAW_CONFIG_HOME", &workspace.config_home)
        .env("HOME", &workspace.home)
        .env("NO_COLOR", "1")
        .args([
            "--model",
            "sonnet",
            "--permission-mode",
            "workspace-write",
            "--allowedTools",
            "write_file",
            "--output-format=json",
        ]);
    configure_quality_process_env(&mut command, workspace);

    let prompt = format!("{SCENARIO_PREFIX}{}", case.name);
    command.arg(prompt);

    let output = command.output().expect("claw should launch");
    if !output.status.success() {
        return Err(output);
    }
    Ok(ScenarioRun {
        response: parse_json_output(&String::from_utf8_lossy(&output.stdout)),
    })
}

fn write_verifier_settings(workspace: &HarnessWorkspace, settings: &Value) {
    fs::write(
        workspace.config_home.join("settings.json"),
        settings.to_string(),
    )
    .expect("settings should write");
}

fn rust_verifier_settings(final_gate: bool) -> Value {
    json!({
        "verifier": {
            "enabled": true,
            "mode": "staged",
            "finalGate": final_gate,
            "cargo": {
                "check": true,
                "clippy": false,
                "fmt": true,
                "test": false,
                "timeoutSecs": 60
            },
            "node": { "enabled": false },
            "python": { "enabled": false }
        }
    })
}

fn node_verifier_settings(final_gate: bool) -> Value {
    json!({
        "verifier": {
            "enabled": true,
            "mode": "staged",
            "finalGate": final_gate,
            "cargo": {
                "check": false,
                "clippy": false,
                "fmt": false,
                "test": false,
                "timeoutSecs": 5
            },
            "node": {
                "enabled": true,
                "timeoutSecs": 10
            },
            "python": { "enabled": false }
        }
    })
}

fn python_verifier_settings(final_gate: bool, timeout_secs: u64) -> Value {
    json!({
        "verifier": {
            "enabled": true,
            "mode": "staged",
            "finalGate": final_gate,
            "cargo": {
                "check": false,
                "clippy": false,
                "fmt": false,
                "test": false,
                "timeoutSecs": 5
            },
            "node": { "enabled": false },
            "python": {
                "enabled": true,
                "timeoutSecs": timeout_secs
            }
        }
    })
}

fn prepare_rust_workspace(workspace: &HarnessWorkspace) {
    fs::create_dir_all(workspace.root.join("crates").join("app").join("src"))
        .expect("rust src dir should exist");
    fs::write(
        workspace.root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/app\"]\nresolver = \"2\"\n",
    )
    .expect("workspace cargo manifest should write");
    fs::write(
        workspace.root.join("crates").join("app").join("Cargo.toml"),
        "[package]\nname = \"demo_app\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("package cargo manifest should write");
    fs::write(
        workspace
            .root
            .join("crates")
            .join("app")
            .join("src")
            .join("lib.rs"),
        "pub fn answer() -> usize {\n    0\n}\n",
    )
    .expect("rust source should write");
}

fn prepare_rust_red_green_fixture(workspace: &HarnessWorkspace) {
    prepare_rust_workspace(workspace);
    write_verifier_settings(workspace, &rust_verifier_settings(true));
}

fn prepare_rust_final_gate_retry_fixture(workspace: &HarnessWorkspace) {
    prepare_rust_workspace(workspace);
    write_verifier_settings(workspace, &rust_verifier_settings(true));
}

fn prepare_rust_config_failure_fixture(workspace: &HarnessWorkspace) {
    fs::create_dir_all(workspace.root.join("src")).expect("rust src dir should exist");
    fs::write(
        workspace.root.join("Cargo.toml"),
        "[package]\nname = \"config_demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("cargo manifest should write");
    fs::write(
        workspace.root.join("src").join("lib.rs"),
        "pub fn answer() -> usize {\n    0\n}\n",
    )
    .expect("rust source should write");
    write_verifier_settings(workspace, &rust_verifier_settings(false));
}

fn node_stub_unix() -> &'static str {
    r#"#!/bin/sh
set -eu
if [ "$#" -lt 2 ] || [ "$1" != "run" ]; then
  echo "unsupported npm invocation: $*" >&2
  exit 2
fi
shift
if [ "$1" = "--silent" ]; then
  shift
fi
script="$1"
content=""
if [ -f "src/index.ts" ]; then
  content="$(cat "src/index.ts")"
fi
case "$content" in
  *TOOL_UNAVAILABLE_SENTINEL*)
    echo "npm not found" >&2
    exit 1
    ;;
esac
case "$script" in
  typecheck)
    case "$content" in
      *BROKEN_TYPECHECK*)
        echo "src/index.ts(1,14): error TS2322: Type 'number' is not assignable to type 'string'." >&2
        exit 1
        ;;
      *)
        exit 0
        ;;
    esac
    ;;
  lint|test)
    exit 0
    ;;
  *)
    echo "unsupported npm script: $script" >&2
    exit 2
    ;;
esac
"#
}

fn node_stub_windows() -> &'static str {
    r#"$arguments = @($args)
if ($arguments.Length -lt 2 -or $arguments[0] -ne 'run') {
    [Console]::Error.WriteLine("unsupported npm invocation: $($arguments -join ' ')")
    exit 2
}
$index = 1
if ($arguments.Length -gt 2 -and $arguments[1] -eq '--silent') {
    $index = 2
}
$script = $arguments[$index]
$sourcePath = Join-Path (Get-Location) 'src/index.ts'
$content = if (Test-Path $sourcePath) { Get-Content -LiteralPath $sourcePath -Raw } else { '' }
if ($content -match 'TOOL_UNAVAILABLE_SENTINEL') {
    [Console]::Error.WriteLine('npm not found')
    exit 1
}
switch ($script) {
    'typecheck' {
        if ($content -match 'BROKEN_TYPECHECK') {
            [Console]::Error.WriteLine("src/index.ts(1,14): error TS2322: Type 'number' is not assignable to type 'string'.")
            exit 1
        }
        exit 0
    }
    'lint' { exit 0 }
    'test' { exit 0 }
    default {
        [Console]::Error.WriteLine("unsupported npm script: $script")
        exit 2
    }
}
"#
}

fn prepare_node_workspace(workspace: &HarnessWorkspace) {
    let package_root = workspace.root.join("packages").join("web");
    fs::create_dir_all(package_root.join("src")).expect("node src dir should exist");
    fs::write(
        package_root.join("package.json"),
        json!({
            "name": "web-app",
            "version": "1.0.0",
            "scripts": {
                "typecheck": "tsc --noEmit",
                "lint": "eslint .",
                "test": "vitest run"
            }
        })
        .to_string(),
    )
    .expect("package.json should write");
    fs::write(
        package_root.join("src").join("index.ts"),
        "export const message = 'seed';\n",
    )
    .expect("node source should write");
    write_program_stub(&workspace.bin, "npm", node_stub_unix(), node_stub_windows());
}

fn prepare_node_red_green_fixture(workspace: &HarnessWorkspace) {
    prepare_node_workspace(workspace);
    write_verifier_settings(workspace, &node_verifier_settings(true));
}

fn prepare_node_tool_unavailable_fixture(workspace: &HarnessWorkspace) {
    prepare_node_workspace(workspace);
    write_verifier_settings(workspace, &node_verifier_settings(false));
}

fn python_stub_unix() -> &'static str {
    r#"#!/bin/sh
set -eu
if [ "$#" -lt 2 ] || [ "$1" != "-m" ]; then
  echo "unsupported python invocation: $*" >&2
  exit 2
fi
module="$2"
content=""
if [ -f "app/main.py" ]; then
  content="$(cat "app/main.py")"
fi
case "$content" in
  *TIMEOUT_SENTINEL*)
    sleep 3
    exit 0
    ;;
esac
case "$module" in
  py_compile)
    case "$content" in
      *BROKEN_PY_COMPILE*)
        echo "SyntaxError: invalid syntax" >&2
        exit 1
        ;;
      *)
        exit 0
        ;;
    esac
    ;;
  pytest)
    exit 0
    ;;
  *)
    echo "unsupported python module: $module" >&2
    exit 2
    ;;
esac
"#
}

fn python_stub_windows() -> &'static str {
    r#"$arguments = @($args)
if ($arguments.Length -lt 2 -or $arguments[0] -ne '-m') {
    [Console]::Error.WriteLine("unsupported python invocation: $($arguments -join ' ')")
    exit 2
}
$module = $arguments[1]
$sourcePath = Join-Path (Get-Location) 'app/main.py'
$content = if (Test-Path $sourcePath) { Get-Content -LiteralPath $sourcePath -Raw } else { '' }
if ($content -match 'TIMEOUT_SENTINEL') {
    Start-Sleep -Seconds 3
    exit 0
}
switch ($module) {
    'py_compile' {
        if ($content -match 'BROKEN_PY_COMPILE') {
            [Console]::Error.WriteLine('SyntaxError: invalid syntax')
            exit 1
        }
        exit 0
    }
    'pytest' { exit 0 }
    default {
        [Console]::Error.WriteLine("unsupported python module: $module")
        exit 2
    }
}
"#
}

fn prepare_python_workspace(workspace: &HarnessWorkspace) {
    let project_root = workspace.root.join("services").join("api");
    fs::create_dir_all(project_root.join("app")).expect("python app dir should exist");
    fs::create_dir_all(project_root.join("tests")).expect("python tests dir should exist");
    fs::write(
        project_root.join("pyproject.toml"),
        "[project]\nname = \"quality-api\"\nversion = \"0.1.0\"\n",
    )
    .expect("pyproject should write");
    fs::write(
        project_root.join("app").join("main.py"),
        "def meaning() -> int:\n    return 1\n",
    )
    .expect("python source should write");
    fs::write(
        project_root.join("tests").join("test_smoke.py"),
        "def test_smoke():\n    assert True\n",
    )
    .expect("python test should write");
    write_program_stub(
        &workspace.bin,
        "python",
        python_stub_unix(),
        python_stub_windows(),
    );
}

fn prepare_python_red_green_fixture(workspace: &HarnessWorkspace) {
    prepare_python_workspace(workspace);
    write_verifier_settings(workspace, &python_verifier_settings(true, 10));
}

fn prepare_python_timeout_fixture(workspace: &HarnessWorkspace) {
    prepare_python_workspace(workspace);
    write_verifier_settings(workspace, &python_verifier_settings(false, 1));
}

fn verification_reports(response: &Value) -> &[Value] {
    response["verification_reports"]
        .as_array()
        .expect("verification reports array")
}

fn report_steps(report: &Value) -> &[Value] {
    report["steps"].as_array().expect("report steps array")
}

fn find_report<'a>(response: &'a Value, phase: &str, status: &str, adapter: &str) -> &'a Value {
    verification_reports(response)
        .iter()
        .find(|report| {
            report["phase"] == phase
                && report["status"] == status
                && report["adapter_id"] == adapter
        })
        .unwrap_or_else(|| {
            panic!(
                "missing report phase={phase} status={status} adapter={adapter}: {}",
                serde_json::to_string_pretty(response).expect("response should serialize")
            )
        })
}

fn has_report_matching<F>(response: &Value, predicate: F) -> bool
where
    F: Fn(&Value) -> bool,
{
    verification_reports(response).iter().any(predicate)
}

fn path_value_ends_with(value: &Value, suffix: &Path) -> bool {
    value
        .as_str()
        .map(PathBuf::from)
        .is_some_and(|path| path.ends_with(suffix))
}

fn assert_rust_red_green(workspace: &HarnessWorkspace, run: &ScenarioRun) {
    assert_eq!(run.response["iterations"], Value::from(3));
    assert_eq!(run.response["tool_uses"].as_array().map(Vec::len), Some(2));
    assert_eq!(
        run.response["tool_results"].as_array().map(Vec::len),
        Some(2)
    );
    assert_eq!(
        run.response["verification_gate"]["attempted"],
        Value::Bool(true)
    );
    assert_eq!(
        run.response["verification_gate"]["passed"],
        Value::Bool(true)
    );

    let quick_failed = find_report(&run.response, "quick", "failed", "rust-cargo");
    assert_eq!(
        quick_failed["primary_failure_kind"],
        Value::String("code".to_string())
    );
    assert!(path_value_ends_with(
        &quick_failed["project_root"],
        Path::new("crates").join("app").as_path()
    ));
    let quick_step = &report_steps(quick_failed)[0];
    assert!(quick_step["command"]
        .as_str()
        .is_some_and(|command| command.contains("cargo check") && command.contains("-p demo_app")));
    assert_eq!(
        quick_step["step_kind"],
        Value::String("cargo_check".to_string())
    );
    assert_eq!(
        quick_step["target_scope"],
        Value::String("package".to_string())
    );
    assert_eq!(
        quick_step["package_name"],
        Value::String("demo_app".to_string())
    );

    let final_passed = find_report(&run.response, "final", "passed", "rust-cargo");
    let final_step = report_steps(final_passed)
        .iter()
        .find(|step| step["step_kind"] == Value::String("cargo_fmt_check".to_string()))
        .expect("cargo fmt step should exist");
    assert_eq!(
        final_step["target_scope"],
        Value::String("workspace".to_string())
    );
    assert_eq!(
        final_step["package_name"],
        Value::String("demo_app".to_string())
    );
    assert!(fs::read_to_string(
        workspace
            .root
            .join("crates")
            .join("app")
            .join("src")
            .join("lib.rs")
    )
    .expect("rust source should exist")
    .contains("42"));
    assert!(run.response["message"]
        .as_str()
        .expect("message text")
        .contains("rust quality red-green complete"));
}

fn assert_node_red_green(workspace: &HarnessWorkspace, run: &ScenarioRun) {
    assert_eq!(run.response["iterations"], Value::from(3));
    assert_eq!(run.response["tool_uses"].as_array().map(Vec::len), Some(2));
    assert_eq!(
        run.response["verification_gate"]["attempted"],
        Value::Bool(true)
    );
    assert_eq!(
        run.response["verification_gate"]["passed"],
        Value::Bool(true)
    );

    let quick_failed = find_report(&run.response, "quick", "failed", "node-typescript");
    assert_eq!(
        quick_failed["primary_failure_kind"],
        Value::String("code".to_string())
    );
    assert!(path_value_ends_with(
        &quick_failed["project_root"],
        Path::new("packages").join("web").as_path()
    ));
    let quick_step = &report_steps(quick_failed)[0];
    assert!(quick_step["command"]
        .as_str()
        .is_some_and(|command| command.contains("npm run --silent typecheck")));
    assert_eq!(
        quick_step["step_kind"],
        Value::String("typecheck".to_string())
    );
    assert_eq!(
        quick_step["target_scope"],
        Value::String("package".to_string())
    );
    assert_eq!(
        quick_step["package_manager"],
        Value::String("npm".to_string())
    );
    assert_eq!(
        quick_step["package_name"],
        Value::String("web-app".to_string())
    );

    let final_passed = find_report(&run.response, "final", "passed", "node-typescript");
    assert!(report_steps(final_passed)
        .iter()
        .any(|step| step["step_kind"] == Value::String("lint".to_string())));
    assert!(report_steps(final_passed)
        .iter()
        .any(|step| step["step_kind"] == Value::String("test".to_string())));
    assert!(fs::read_to_string(
        workspace
            .root
            .join("packages")
            .join("web")
            .join("src")
            .join("index.ts")
    )
    .expect("node source should exist")
    .contains("\"ok\""));
    assert!(run.response["message"]
        .as_str()
        .expect("message text")
        .contains("node quality red-green complete"));
}

fn assert_python_red_green(workspace: &HarnessWorkspace, run: &ScenarioRun) {
    assert_eq!(run.response["iterations"], Value::from(3));
    assert_eq!(run.response["tool_uses"].as_array().map(Vec::len), Some(2));
    assert_eq!(
        run.response["verification_gate"]["attempted"],
        Value::Bool(true)
    );
    assert_eq!(
        run.response["verification_gate"]["passed"],
        Value::Bool(true)
    );

    let quick_failed = find_report(&run.response, "quick", "failed", "python");
    assert_eq!(
        quick_failed["primary_failure_kind"],
        Value::String("code".to_string())
    );
    assert!(path_value_ends_with(
        &quick_failed["project_root"],
        Path::new("services").join("api").as_path()
    ));
    let quick_step = &report_steps(quick_failed)[0];
    assert_eq!(
        quick_step["step_kind"],
        Value::String("py_compile".to_string())
    );
    assert_eq!(
        quick_step["target_scope"],
        Value::String("file_set".to_string())
    );
    assert_eq!(
        quick_step["launcher_kind"],
        Value::String("global".to_string())
    );

    let final_passed = find_report(&run.response, "final", "passed", "python");
    assert!(report_steps(final_passed)
        .iter()
        .any(|step| step["step_kind"] == Value::String("pytest".to_string())));
    assert!(fs::read_to_string(
        workspace
            .root
            .join("services")
            .join("api")
            .join("app")
            .join("main.py")
    )
    .expect("python source should exist")
    .contains("return 42"));
    assert!(run.response["message"]
        .as_str()
        .expect("message text")
        .contains("python quality red-green complete"));
}

fn assert_rust_config_failure(_: &HarnessWorkspace, run: &ScenarioRun) {
    assert_eq!(run.response["iterations"], Value::from(2));
    assert_eq!(run.response["tool_uses"].as_array().map(Vec::len), Some(1));
    assert_eq!(
        run.response["verification_gate"]["attempted"],
        Value::Bool(false)
    );
    assert_eq!(
        run.response["verification_gate"]["passed"],
        Value::Bool(true)
    );

    let quick_failed = find_report(&run.response, "quick", "failed", "rust-cargo");
    assert_eq!(
        quick_failed["primary_failure_kind"],
        Value::String("config".to_string())
    );
    let quick_step = &report_steps(quick_failed)[0];
    assert_eq!(
        quick_step["failure_kind"],
        Value::String("config".to_string())
    );
    assert!(run.response["message"]
        .as_str()
        .expect("message text")
        .contains("rust config failure captured"));
}

fn assert_node_tool_unavailable(_: &HarnessWorkspace, run: &ScenarioRun) {
    assert_eq!(run.response["iterations"], Value::from(2));
    assert_eq!(
        run.response["verification_gate"]["attempted"],
        Value::Bool(false)
    );
    let quick_failed = find_report(&run.response, "quick", "failed", "node-typescript");
    assert_eq!(
        quick_failed["primary_failure_kind"],
        Value::String("tool_unavailable".to_string())
    );
    let quick_step = &report_steps(quick_failed)[0];
    assert_eq!(
        quick_step["failure_kind"],
        Value::String("tool_unavailable".to_string())
    );
    assert_eq!(
        quick_step["step_kind"],
        Value::String("typecheck".to_string())
    );
    assert_eq!(
        quick_step["package_manager"],
        Value::String("npm".to_string())
    );
    assert!(run.response["message"]
        .as_str()
        .expect("message text")
        .contains("node tool unavailable captured"));
}

fn assert_python_timeout(_: &HarnessWorkspace, run: &ScenarioRun) {
    assert_eq!(run.response["iterations"], Value::from(2));
    assert_eq!(
        run.response["verification_gate"]["attempted"],
        Value::Bool(false)
    );
    let quick_failed = find_report(&run.response, "quick", "failed", "python");
    assert_eq!(
        quick_failed["primary_failure_kind"],
        Value::String("timeout".to_string())
    );
    let quick_step = &report_steps(quick_failed)[0];
    assert_eq!(
        quick_step["failure_kind"],
        Value::String("timeout".to_string())
    );
    assert_eq!(
        quick_step["step_kind"],
        Value::String("py_compile".to_string())
    );
    assert_eq!(
        quick_step["launcher_kind"],
        Value::String("global".to_string())
    );
    assert!(run.response["message"]
        .as_str()
        .expect("message text")
        .contains("python timeout captured"));
}

fn assert_rust_final_gate_retry(workspace: &HarnessWorkspace, run: &ScenarioRun) {
    assert!(
        run.response["iterations"]
            .as_u64()
            .is_some_and(|iterations| iterations >= 5),
        "expected retry flow to require at least five iterations: {}",
        serde_json::to_string_pretty(&run.response).expect("response should serialize")
    );
    assert_eq!(run.response["tool_uses"].as_array().map(Vec::len), Some(2));
    assert_eq!(
        run.response["verification_gate"]["attempted"],
        Value::Bool(true)
    );
    assert_eq!(
        run.response["verification_gate"]["passed"],
        Value::Bool(true)
    );

    assert!(has_report_matching(&run.response, |report| {
        report["phase"] == "final"
            && report["status"] == "failed"
            && report["adapter_id"] == "rust-cargo"
            && !report_steps(report).is_empty()
    }));
    assert!(has_report_matching(&run.response, |report| {
        report["phase"] == "final"
            && report["adapter_id"] == "rust-cargo"
            && report_steps(report).is_empty()
            && report["summary_text"]
                .as_str()
                .is_some_and(|summary| summary.contains("still failing"))
    }));
    assert!(has_report_matching(&run.response, |report| {
        report["phase"] == "final"
            && report["status"] == "passed"
            && report["adapter_id"] == "rust-cargo"
    }));

    let final_failed = find_report(&run.response, "final", "failed", "rust-cargo");
    let failing_fmt_step = report_steps(final_failed)
        .iter()
        .find(|step| step["step_kind"] == Value::String("cargo_fmt_check".to_string()))
        .expect("failing cargo fmt step should exist");
    assert_eq!(
        failing_fmt_step["target_scope"],
        Value::String("workspace".to_string())
    );
    assert_eq!(
        failing_fmt_step["package_name"],
        Value::String("demo_app".to_string())
    );
    assert!(fs::read_to_string(
        workspace
            .root
            .join("crates")
            .join("app")
            .join("src")
            .join("lib.rs")
    )
    .expect("rust source should exist")
    .contains("pub fn answer() -> usize"));
    assert!(run.response["message"]
        .as_str()
        .expect("message text")
        .contains("rust final gate retry complete"));
}

fn parse_json_output(stdout: &str) -> Value {
    stdout
        .lines()
        .rev()
        .find_map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with('{') && trimmed.ends_with('}') {
                serde_json::from_str(trimmed).ok()
            } else {
                None
            }
        })
        .unwrap_or_else(|| panic!("no JSON response line found in stdout:\n{stdout}"))
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after epoch")
        .as_millis();
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "claw-mock-quality-{label}-{}-{millis}-{counter}",
        std::process::id()
    ))
}
