//! Structured self-verification of code edits.
//!
//! The verifier consumes successful write/edit tool invocations, detects the
//! owning project for the touched file, and runs phase-aware validation. The
//! runtime uses the resulting structured reports both as model-visible
//! feedback and as a final gate before ending a turn in staged mode.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::Value;
use toml::Value as TomlValue;

/// Maximum bytes kept per verification step before the summary is truncated.
const DEFAULT_MAX_OUTPUT_BYTES: usize = 2_048;
const WRITE_TOOLS: &[&str] = &["edit_file", "write_file", "Edit", "Write"];
const ESLINT_CONFIG_FILES: &[&str] = &[
    "eslint.config.js",
    "eslint.config.cjs",
    "eslint.config.mjs",
    ".eslintrc",
    ".eslintrc.js",
    ".eslintrc.cjs",
    ".eslintrc.json",
    ".eslintrc.yml",
    ".eslintrc.yaml",
];
const PYTHON_ROOT_MARKERS: &[&str] = &[
    "pyproject.toml",
    "uv.lock",
    "poetry.lock",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "setup.py",
    "setup.cfg",
    "tox.ini",
];
const RUFF_CONFIG_FILES: &[&str] = &["ruff.toml", ".ruff.toml"];
const MYPY_CONFIG_FILES: &[&str] = &["mypy.ini", ".mypy.ini"];
const PYTEST_CONFIG_FILES: &[&str] = &["pytest.ini", "tox.ini", "setup.cfg"];
static REPORT_COUNTER: AtomicU64 = AtomicU64::new(1);

/// High-level phase for a verification run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationPhase {
    Quick,
    Final,
}

impl VerificationPhase {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Final => "final",
        }
    }
}

/// Outcome classification for a verification step or report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationStatus {
    Passed,
    Failed,
    Skipped,
    Unavailable,
}

/// Refined failure classification used by structured verification steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationFailureKind {
    Code,
    Environment,
    ToolUnavailable,
    Config,
    Timeout,
}

impl VerificationFailureKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Code => "code",
            Self::Environment => "environment",
            Self::ToolUnavailable => "tool_unavailable",
            Self::Config => "config",
            Self::Timeout => "timeout",
        }
    }
}

impl VerificationStatus {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Passed => "passed",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
            Self::Unavailable => "unavailable",
        }
    }

    #[must_use]
    pub fn is_success(self) -> bool {
        matches!(self, Self::Passed | Self::Skipped)
    }
}

/// Mutable-work context supplied by the runtime for a verification decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationContext {
    pub phase: VerificationPhase,
    pub workspace_root: Option<PathBuf>,
    pub tool_name: String,
    pub tool_input: String,
    pub touched_paths: Vec<PathBuf>,
    pub mutation_sequence: u64,
}

impl VerificationContext {
    #[must_use]
    pub fn from_tool_invocation(
        phase: VerificationPhase,
        workspace_root: Option<PathBuf>,
        tool_name: impl Into<String>,
        tool_input: impl Into<String>,
        mutation_sequence: u64,
    ) -> Option<Self> {
        let tool_name = tool_name.into();
        let tool_input = tool_input.into();
        let touched_path = extract_file_path(&tool_input)?;
        Some(Self {
            phase,
            workspace_root,
            tool_name,
            tool_input,
            touched_paths: vec![touched_path],
            mutation_sequence,
        })
    }
}

/// Deduplicated project target used by staged final-gate verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationTarget {
    pub adapter_id: String,
    pub project_root: PathBuf,
    pub touched_paths: Vec<PathBuf>,
    pub mutation_sequence: u64,
}

/// Structured output of one verification step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationStepReport {
    pub adapter: String,
    pub project_root: PathBuf,
    pub label: String,
    pub command: String,
    pub phase: VerificationPhase,
    pub status: VerificationStatus,
    pub failure_kind: Option<VerificationFailureKind>,
    pub duration_ms: u64,
    pub truncated_output: String,
}

/// Structured output of a full verification pass for one adapter/root pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationReport {
    pub report_id: String,
    pub phase: VerificationPhase,
    pub adapter_id: String,
    pub project_root: PathBuf,
    pub touched_paths: Vec<PathBuf>,
    pub status: VerificationStatus,
    pub summary_text: String,
    pub steps: Vec<VerificationStepReport>,
}

impl VerificationReport {
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }

    #[must_use]
    pub fn short_summary(&self) -> String {
        let mut lines = self.summary_text.lines();
        let first = lines.next().unwrap_or_default().trim();
        if first.is_empty() {
            format!(
                "[verifier:{}:{}] {}",
                self.phase.as_str(),
                self.adapter_id,
                self.status.as_str()
            )
        } else {
            first.to_string()
        }
    }
}

/// Status of the staged final gate for the completed turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationGateStatus {
    pub attempted: bool,
    pub passed: bool,
    pub report_ids: Vec<String>,
}

impl VerificationGateStatus {
    #[must_use]
    pub fn not_required() -> Self {
        Self {
            attempted: false,
            passed: true,
            report_ids: Vec::new(),
        }
    }
}

/// Strategy that inspects completed mutations and produces verification
/// reports for the runtime.
pub trait Verifier: Send {
    fn quick_verify(&self, context: &VerificationContext) -> Vec<VerificationReport>;
    fn final_verify(&self, target: &VerificationTarget) -> Option<VerificationReport>;
    fn final_gate_enabled(&self) -> bool {
        false
    }
}

/// Declarative runtime config for the built-in multi-language verifier.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CargoVerifierConfig {
    pub legacy_mode: bool,
    pub quick_on_write: bool,
    pub final_gate: bool,
    pub max_output_bytes: usize,
    pub rust_check: bool,
    pub rust_clippy: bool,
    pub rust_fmt: bool,
    pub rust_test: bool,
    pub rust_timeout: Duration,
    pub node_enabled: bool,
    pub node_timeout: Duration,
    pub python_enabled: bool,
    pub python_timeout: Duration,
}

impl Default for CargoVerifierConfig {
    fn default() -> Self {
        Self {
            legacy_mode: true,
            quick_on_write: true,
            final_gate: false,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            rust_check: true,
            rust_clippy: true,
            rust_fmt: true,
            rust_test: true,
            rust_timeout: Duration::from_mins(2),
            node_enabled: true,
            node_timeout: Duration::from_mins(2),
            python_enabled: true,
            python_timeout: Duration::from_mins(2),
        }
    }
}

/// Built-in verifier registry for Rust, Node/TypeScript, and Python roots.
pub struct CargoVerifier {
    config: CargoVerifierConfig,
}

impl CargoVerifier {
    #[must_use]
    pub fn new(config: CargoVerifierConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub fn final_gate_enabled(&self) -> bool {
        !self.config.legacy_mode && self.config.final_gate
    }
}

impl Verifier for CargoVerifier {
    fn quick_verify(&self, context: &VerificationContext) -> Vec<VerificationReport> {
        if !WRITE_TOOLS.contains(&context.tool_name.as_str()) {
            return Vec::new();
        }
        let Some(path) = context.touched_paths.first() else {
            return Vec::new();
        };

        for adapter in [Adapter::Rust, Adapter::NodeTypeScript, Adapter::Python] {
            if let Some(report) = adapter.quick_verify(path, context, &self.config) {
                return vec![report];
            }
        }

        Vec::new()
    }

    fn final_verify(&self, target: &VerificationTarget) -> Option<VerificationReport> {
        if self.config.legacy_mode || !self.config.final_gate {
            return None;
        }
        let adapter = Adapter::by_id(&target.adapter_id)?;
        adapter.final_verify(target, &self.config)
    }

    fn final_gate_enabled(&self) -> bool {
        !self.config.legacy_mode && self.config.final_gate
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Adapter {
    Rust,
    NodeTypeScript,
    Python,
}

impl Adapter {
    fn by_id(value: &str) -> Option<Self> {
        match value {
            "rust-cargo" => Some(Self::Rust),
            "node-typescript" => Some(Self::NodeTypeScript),
            "python" => Some(Self::Python),
            _ => None,
        }
    }

    fn quick_verify(
        self,
        path: &Path,
        context: &VerificationContext,
        config: &CargoVerifierConfig,
    ) -> Option<VerificationReport> {
        match self {
            Self::Rust => verify_rust(path, context, config),
            Self::NodeTypeScript => verify_node(path, context, config),
            Self::Python => verify_python(path, context, config),
        }
    }

    fn final_verify(
        self,
        target: &VerificationTarget,
        config: &CargoVerifierConfig,
    ) -> Option<VerificationReport> {
        match self {
            Self::Rust => Some(finalize_rust(target, config)),
            Self::NodeTypeScript => finalize_node(target, config),
            Self::Python => finalize_python(target, config),
        }
    }
}

#[derive(Debug, Clone)]
struct PlannedStep {
    label: String,
    command: Vec<String>,
    diagnostics: StepDiagnostics,
}

#[derive(Debug)]
enum StepOutcome {
    Passed {
        body: String,
        duration_ms: u64,
    },
    Failed {
        body: String,
        duration_ms: u64,
        failure_kind: Option<VerificationFailureKind>,
    },
    Unavailable {
        message: String,
        duration_ms: u64,
        failure_kind: Option<VerificationFailureKind>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackageManager {
    Npm,
    Pnpm,
    Yarn,
    Bun,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PythonLauncherKind {
    Uv,
    Poetry,
    Venv,
    Global,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PythonRunner {
    command_prefix: Vec<String>,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct PythonProjectProfile {
    project_root: PathBuf,
    runner: PythonRunner,
    launcher_kind: PythonLauncherKind,
    pyproject_parsed: bool,
    has_ruff: bool,
    has_mypy: bool,
    has_pytest: bool,
    typed_targets: Vec<PathBuf>,
    test_root_present: bool,
    pyproject_path: Option<PathBuf>,
    pyproject_parse_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PythonStepKind {
    RuffCheck,
    Mypy,
    Pytest,
    PyCompile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepDiagnostics {
    Generic,
    Python {
        launcher_kind: PythonLauncherKind,
        step_kind: PythonStepKind,
    },
}

fn verify_rust(
    path: &Path,
    context: &VerificationContext,
    config: &CargoVerifierConfig,
) -> Option<VerificationReport> {
    let manifest = nearest_file(path, "Cargo.toml")?;
    let project_root = manifest.parent()?.to_path_buf();
    let phase = context.phase;
    let steps = if config.legacy_mode {
        rust_legacy_steps(config)
    } else if phase == VerificationPhase::Quick {
        rust_quick_steps(config)
    } else {
        rust_final_steps(config)
    };
    Some(run_rust_steps(
        &project_root,
        context.touched_paths.clone(),
        phase,
        steps,
        config,
    ))
}

fn finalize_rust(target: &VerificationTarget, config: &CargoVerifierConfig) -> VerificationReport {
    let steps = if config.legacy_mode {
        rust_legacy_steps(config)
    } else {
        rust_final_steps(config)
    };
    run_rust_steps(
        &target.project_root,
        target.touched_paths.clone(),
        VerificationPhase::Final,
        steps,
        config,
    )
}

fn run_rust_steps(
    project_root: &Path,
    touched_paths: Vec<PathBuf>,
    phase: VerificationPhase,
    steps: Vec<PlannedStep>,
    config: &CargoVerifierConfig,
) -> VerificationReport {
    run_planned_steps(
        "rust-cargo",
        project_root,
        touched_paths,
        phase,
        steps,
        config.rust_timeout,
        config.max_output_bytes,
    )
}

fn rust_quick_steps(config: &CargoVerifierConfig) -> Vec<PlannedStep> {
    if config.quick_on_write && config.rust_check {
        vec![PlannedStep {
            label: "cargo check".to_string(),
            command: vec![
                "cargo".to_string(),
                "check".to_string(),
                "--quiet".to_string(),
                "--message-format=short".to_string(),
            ],
            diagnostics: StepDiagnostics::Generic,
        }]
    } else {
        Vec::new()
    }
}

fn rust_final_steps(config: &CargoVerifierConfig) -> Vec<PlannedStep> {
    let mut steps = Vec::new();
    if config.rust_fmt {
        steps.push(PlannedStep {
            label: "cargo fmt --check".to_string(),
            command: vec![
                "cargo".to_string(),
                "fmt".to_string(),
                "--".to_string(),
                "--check".to_string(),
            ],
            diagnostics: StepDiagnostics::Generic,
        });
    }
    if config.rust_clippy {
        steps.push(PlannedStep {
            label: "cargo clippy".to_string(),
            command: vec![
                "cargo".to_string(),
                "clippy".to_string(),
                "--quiet".to_string(),
                "--message-format=short".to_string(),
                "--".to_string(),
                "-D".to_string(),
                "warnings".to_string(),
            ],
            diagnostics: StepDiagnostics::Generic,
        });
    }
    if config.rust_test {
        steps.push(PlannedStep {
            label: "cargo test".to_string(),
            command: vec![
                "cargo".to_string(),
                "test".to_string(),
                "--quiet".to_string(),
                "--no-fail-fast".to_string(),
            ],
            diagnostics: StepDiagnostics::Generic,
        });
    }
    steps
}

fn rust_legacy_steps(config: &CargoVerifierConfig) -> Vec<PlannedStep> {
    let mut steps = rust_quick_steps(config);
    if config.rust_clippy {
        steps.push(PlannedStep {
            label: "cargo clippy".to_string(),
            command: vec![
                "cargo".to_string(),
                "clippy".to_string(),
                "--quiet".to_string(),
                "--message-format=short".to_string(),
                "--".to_string(),
                "-D".to_string(),
                "warnings".to_string(),
            ],
            diagnostics: StepDiagnostics::Generic,
        });
    }
    if config.rust_fmt {
        steps.push(PlannedStep {
            label: "cargo fmt --check".to_string(),
            command: vec![
                "cargo".to_string(),
                "fmt".to_string(),
                "--".to_string(),
                "--check".to_string(),
            ],
            diagnostics: StepDiagnostics::Generic,
        });
    }
    if config.rust_test {
        steps.push(PlannedStep {
            label: "cargo test".to_string(),
            command: vec![
                "cargo".to_string(),
                "test".to_string(),
                "--quiet".to_string(),
                "--no-fail-fast".to_string(),
            ],
            diagnostics: StepDiagnostics::Generic,
        });
    }
    steps
}

fn verify_node(
    path: &Path,
    context: &VerificationContext,
    config: &CargoVerifierConfig,
) -> Option<VerificationReport> {
    if !config.node_enabled {
        return None;
    }
    let package_json = nearest_file(path, "package.json")?;
    let project_root = package_json.parent()?.to_path_buf();
    let package_value = match load_node_package(&package_json) {
        Ok(value) => value,
        Err(report) => {
            return Some(node_setup_failure_report(
                &project_root,
                context.touched_paths.clone(),
                context.phase,
                &report,
                config.max_output_bytes,
            ));
        }
    };
    let phase = context.phase;
    let package_manager = detect_package_manager(&project_root);
    let steps = if config.legacy_mode {
        node_legacy_steps(&project_root, &package_value, package_manager)
    } else if phase == VerificationPhase::Quick {
        if config.quick_on_write {
            node_quick_steps(&project_root, &package_value, package_manager)
        } else {
            Vec::new()
        }
    } else {
        node_final_steps(&project_root, &package_value, package_manager)
    };
    Some(run_planned_steps(
        "node-typescript",
        &project_root,
        context.touched_paths.clone(),
        phase,
        steps,
        config.node_timeout,
        config.max_output_bytes,
    ))
}

fn finalize_node(
    target: &VerificationTarget,
    config: &CargoVerifierConfig,
) -> Option<VerificationReport> {
    if !config.node_enabled {
        return None;
    }
    let package_json = target.project_root.join("package.json");
    let package_value = match load_node_package(&package_json) {
        Ok(value) => value,
        Err(report) => {
            return Some(node_setup_failure_report(
                &target.project_root,
                target.touched_paths.clone(),
                VerificationPhase::Final,
                &report,
                config.max_output_bytes,
            ));
        }
    };
    let package_manager = detect_package_manager(&target.project_root);
    Some(run_planned_steps(
        "node-typescript",
        &target.project_root,
        target.touched_paths.clone(),
        VerificationPhase::Final,
        node_final_steps(&target.project_root, &package_value, package_manager),
        config.node_timeout,
        config.max_output_bytes,
    ))
}

struct NodeSetupFailure {
    label: String,
    kind: VerificationFailureKind,
    message: String,
}

fn load_node_package(package_json: &Path) -> Result<Value, NodeSetupFailure> {
    let contents = fs::read_to_string(package_json).map_err(|error| NodeSetupFailure {
        label: "package.json read".to_string(),
        kind: VerificationFailureKind::Environment,
        message: format!("failed to read {}: {error}", package_json.display()),
    })?;
    serde_json::from_str::<Value>(&contents).map_err(|error| NodeSetupFailure {
        label: "package.json parse".to_string(),
        kind: VerificationFailureKind::Config,
        message: format!("failed to parse {}: {error}", package_json.display()),
    })
}

fn node_setup_failure_report(
    project_root: &Path,
    touched_paths: Vec<PathBuf>,
    phase: VerificationPhase,
    failure: &NodeSetupFailure,
    max_output_bytes: usize,
) -> VerificationReport {
    let steps = vec![VerificationStepReport {
        adapter: "node-typescript".to_string(),
        project_root: project_root.to_path_buf(),
        label: failure.label.clone(),
        command: project_root.join("package.json").display().to_string(),
        phase,
        status: VerificationStatus::Unavailable,
        failure_kind: Some(failure.kind),
        duration_ms: 0,
        truncated_output: truncate_output(&failure.message, max_output_bytes),
    }];
    let summary_text = render_report_summary(
        "node-typescript",
        project_root,
        phase,
        VerificationStatus::Unavailable,
        &steps,
    );
    VerificationReport {
        report_id: next_report_id(),
        phase,
        adapter_id: "node-typescript".to_string(),
        project_root: project_root.to_path_buf(),
        touched_paths,
        status: VerificationStatus::Unavailable,
        summary_text,
        steps,
    }
}

fn node_quick_steps(
    root: &Path,
    package_value: &Value,
    manager: PackageManager,
) -> Vec<PlannedStep> {
    if has_script(package_value, "typecheck") {
        return vec![PlannedStep {
            label: "typecheck".to_string(),
            command: package_manager_run_script(manager, "typecheck"),
            diagnostics: StepDiagnostics::Generic,
        }];
    }
    if root.join("tsconfig.json").is_file() {
        return vec![PlannedStep {
            label: "tsc --noEmit".to_string(),
            command: package_manager_exec(manager, "tsc", &["--noEmit"]),
            diagnostics: StepDiagnostics::Generic,
        }];
    }
    Vec::new()
}

fn node_final_steps(
    root: &Path,
    package_value: &Value,
    manager: PackageManager,
) -> Vec<PlannedStep> {
    let mut steps = Vec::new();
    if has_script(package_value, "lint") {
        steps.push(PlannedStep {
            label: "lint".to_string(),
            command: package_manager_run_script(manager, "lint"),
            diagnostics: StepDiagnostics::Generic,
        });
    } else if ESLINT_CONFIG_FILES
        .iter()
        .any(|name| root.join(name).is_file())
    {
        steps.push(PlannedStep {
            label: "eslint .".to_string(),
            command: package_manager_exec(manager, "eslint", &["."]),
            diagnostics: StepDiagnostics::Generic,
        });
    }
    if has_script(package_value, "test") {
        steps.push(PlannedStep {
            label: "test".to_string(),
            command: package_manager_run_script(manager, "test"),
            diagnostics: StepDiagnostics::Generic,
        });
    }
    steps
}

fn node_legacy_steps(
    root: &Path,
    package_value: &Value,
    manager: PackageManager,
) -> Vec<PlannedStep> {
    let mut steps = node_quick_steps(root, package_value, manager);
    steps.extend(node_final_steps(root, package_value, manager));
    steps
}

fn verify_python(
    path: &Path,
    context: &VerificationContext,
    config: &CargoVerifierConfig,
) -> Option<VerificationReport> {
    verify_python_for_phase(path, &context.touched_paths, context.phase, config)
}

fn finalize_python(
    target: &VerificationTarget,
    config: &CargoVerifierConfig,
) -> Option<VerificationReport> {
    if !config.python_enabled {
        return None;
    }
    let profile = build_python_profile_for_root(&target.project_root)?;
    Some(build_python_report(
        &profile,
        target.touched_paths.clone(),
        VerificationPhase::Final,
        config,
    ))
}

fn verify_python_for_phase(
    path: &Path,
    touched_paths: &[PathBuf],
    phase: VerificationPhase,
    config: &CargoVerifierConfig,
) -> Option<VerificationReport> {
    if !config.python_enabled {
        return None;
    }
    let profile = build_python_profile_for_path(path)?;
    Some(build_python_report(
        &profile,
        touched_paths.to_vec(),
        phase,
        config,
    ))
}

fn build_python_report(
    profile: &PythonProjectProfile,
    touched_paths: Vec<PathBuf>,
    phase: VerificationPhase,
    config: &CargoVerifierConfig,
) -> VerificationReport {
    if let Some(report) = python_config_failure_report(
        profile,
        phase,
        touched_paths.clone(),
        config.max_output_bytes,
    ) {
        return report;
    }

    let steps = if config.legacy_mode {
        python_legacy_steps(profile, &touched_paths)
    } else if phase == VerificationPhase::Quick {
        if config.quick_on_write {
            python_quick_steps(profile, &touched_paths)
        } else {
            Vec::new()
        }
    } else {
        python_final_steps(profile, &touched_paths)
    };

    run_planned_steps(
        "python",
        &profile.project_root,
        touched_paths,
        phase,
        steps,
        config.python_timeout,
        config.max_output_bytes,
    )
}

fn python_config_failure_report(
    profile: &PythonProjectProfile,
    phase: VerificationPhase,
    touched_paths: Vec<PathBuf>,
    max_output_bytes: usize,
) -> Option<VerificationReport> {
    if profile.pyproject_parsed {
        return None;
    }
    let error = profile.pyproject_parse_error.as_ref()?;
    let pyproject_path = profile.pyproject_path.as_ref().map_or_else(
        || "pyproject.toml".to_string(),
        |path| path.display().to_string(),
    );
    let steps = vec![VerificationStepReport {
        adapter: "python".to_string(),
        project_root: profile.project_root.clone(),
        label: "pyproject.toml parse".to_string(),
        command: pyproject_path,
        phase,
        status: VerificationStatus::Failed,
        failure_kind: Some(VerificationFailureKind::Config),
        duration_ms: 0,
        truncated_output: truncate_output(error, max_output_bytes),
    }];
    let summary_text = render_report_summary(
        "python",
        &profile.project_root,
        phase,
        VerificationStatus::Failed,
        &steps,
    );
    Some(VerificationReport {
        report_id: next_report_id(),
        phase,
        adapter_id: "python".to_string(),
        project_root: profile.project_root.clone(),
        touched_paths,
        status: VerificationStatus::Failed,
        summary_text,
        steps,
    })
}

fn python_quick_steps(
    profile: &PythonProjectProfile,
    touched_paths: &[PathBuf],
) -> Vec<PlannedStep> {
    let python_files = python_source_targets(&profile.project_root, touched_paths);
    if profile.has_ruff && !python_files.is_empty() {
        return vec![python_step(
            profile,
            PythonStepKind::RuffCheck,
            "ruff check",
            python_module_command(&profile.runner, "ruff", &["check"], &python_files),
        )];
    }
    if profile.has_mypy {
        let targets = derive_mypy_targets_from_touched(&profile.project_root, touched_paths);
        if !targets.is_empty() {
            return vec![python_step(
                profile,
                PythonStepKind::Mypy,
                "mypy",
                python_module_command(&profile.runner, "mypy", &[], &targets),
            )];
        }
    }
    if python_files.is_empty() {
        return Vec::new();
    }
    vec![python_step(
        profile,
        PythonStepKind::PyCompile,
        "python -m py_compile",
        python_module_command(&profile.runner, "py_compile", &[], &python_files),
    )]
}

fn python_final_steps(
    profile: &PythonProjectProfile,
    touched_paths: &[PathBuf],
) -> Vec<PlannedStep> {
    let mut steps = Vec::new();
    debug_assert_eq!(
        profile.test_root_present,
        profile.project_root.join("tests").is_dir()
    );
    if profile.has_ruff {
        steps.push(python_step(
            profile,
            PythonStepKind::RuffCheck,
            "ruff check",
            python_module_command(
                &profile.runner,
                "ruff",
                &["check"],
                std::slice::from_ref(&profile.project_root),
            ),
        ));
    }
    if profile.has_mypy {
        let targets = if profile.typed_targets.is_empty() {
            let derived = derive_mypy_targets_from_touched(&profile.project_root, touched_paths);
            if derived.is_empty() {
                vec![profile.project_root.clone()]
            } else {
                derived
            }
        } else {
            profile.typed_targets.clone()
        };
        steps.push(python_step(
            profile,
            PythonStepKind::Mypy,
            "mypy",
            python_module_command(&profile.runner, "mypy", &[], &targets),
        ));
    }
    if profile.has_pytest {
        steps.push(python_step(
            profile,
            PythonStepKind::Pytest,
            "pytest",
            python_module_command(
                &profile.runner,
                "pytest",
                &[],
                std::slice::from_ref(&profile.project_root),
            ),
        ));
    }
    steps
}

fn python_legacy_steps(
    profile: &PythonProjectProfile,
    touched_paths: &[PathBuf],
) -> Vec<PlannedStep> {
    let mut steps = python_quick_steps(profile, touched_paths);
    steps.extend(python_final_steps(profile, touched_paths));
    dedupe_steps(&mut steps);
    steps
}

fn python_step(
    profile: &PythonProjectProfile,
    step_kind: PythonStepKind,
    label: &str,
    command: Vec<String>,
) -> PlannedStep {
    PlannedStep {
        label: label.to_string(),
        command,
        diagnostics: StepDiagnostics::Python {
            launcher_kind: profile.launcher_kind,
            step_kind,
        },
    }
}

fn dedupe_steps(steps: &mut Vec<PlannedStep>) {
    let mut seen = Vec::<String>::new();
    steps.retain(|step| {
        let key = step.label.clone();
        if seen.contains(&key) {
            false
        } else {
            seen.push(key);
            true
        }
    });
}

fn run_planned_steps(
    adapter_id: &str,
    project_root: &Path,
    touched_paths: Vec<PathBuf>,
    phase: VerificationPhase,
    steps: Vec<PlannedStep>,
    timeout: Duration,
    max_output_bytes: usize,
) -> VerificationReport {
    let mut reports = Vec::new();
    let mut report_status = if steps.is_empty() {
        VerificationStatus::Skipped
    } else {
        VerificationStatus::Passed
    };
    let mut skip_remaining = false;

    for step in steps {
        if skip_remaining {
            reports.push(VerificationStepReport {
                adapter: adapter_id.to_string(),
                project_root: project_root.to_path_buf(),
                label: step.label.clone(),
                command: step.command.join(" "),
                phase,
                status: VerificationStatus::Skipped,
                failure_kind: None,
                duration_ms: 0,
                truncated_output: String::new(),
            });
            continue;
        }

        let outcome = run_step(project_root, &step, timeout, max_output_bytes);
        match outcome {
            StepOutcome::Passed { body, duration_ms } => reports.push(VerificationStepReport {
                adapter: adapter_id.to_string(),
                project_root: project_root.to_path_buf(),
                label: step.label.clone(),
                command: step.command.join(" "),
                phase,
                status: VerificationStatus::Passed,
                failure_kind: None,
                duration_ms,
                truncated_output: body,
            }),
            StepOutcome::Failed {
                body,
                duration_ms,
                failure_kind,
            } => {
                report_status = VerificationStatus::Failed;
                skip_remaining = true;
                reports.push(VerificationStepReport {
                    adapter: adapter_id.to_string(),
                    project_root: project_root.to_path_buf(),
                    label: step.label.clone(),
                    command: step.command.join(" "),
                    phase,
                    status: VerificationStatus::Failed,
                    failure_kind,
                    duration_ms,
                    truncated_output: body,
                });
            }
            StepOutcome::Unavailable {
                message,
                duration_ms,
                failure_kind,
            } => {
                report_status = VerificationStatus::Unavailable;
                skip_remaining = true;
                reports.push(VerificationStepReport {
                    adapter: adapter_id.to_string(),
                    project_root: project_root.to_path_buf(),
                    label: step.label.clone(),
                    command: step.command.join(" "),
                    phase,
                    status: VerificationStatus::Unavailable,
                    failure_kind,
                    duration_ms,
                    truncated_output: message,
                });
            }
        }
    }

    let summary_text =
        render_report_summary(adapter_id, project_root, phase, report_status, &reports);
    VerificationReport {
        report_id: next_report_id(),
        phase,
        adapter_id: adapter_id.to_string(),
        project_root: project_root.to_path_buf(),
        touched_paths,
        status: report_status,
        summary_text,
        steps: reports,
    }
}

fn render_report_summary(
    adapter_id: &str,
    project_root: &Path,
    phase: VerificationPhase,
    status: VerificationStatus,
    steps: &[VerificationStepReport],
) -> String {
    if adapter_id == "python" {
        return render_python_report_summary(adapter_id, project_root, phase, status, steps);
    }

    let mut summary = format!(
        "[verifier:{}:{}] {} ({})",
        phase.as_str(),
        adapter_id,
        status.as_str(),
        project_root.display()
    );
    if steps.is_empty() {
        summary.push_str("\n[verifier] no verification steps were planned");
        return summary;
    }
    for step in steps {
        let label = match step.status {
            VerificationStatus::Passed => "ok",
            VerificationStatus::Failed => "FAIL",
            VerificationStatus::Skipped => "skipped",
            VerificationStatus::Unavailable => "unavailable",
        };
        let _ = writeln!(summary, "\n[verifier] {}: {label}", step.label);
        if !step.truncated_output.trim().is_empty() {
            summary.push_str(&step.truncated_output);
        }
    }
    summary.trim_end().to_string()
}

fn render_python_report_summary(
    adapter_id: &str,
    project_root: &Path,
    phase: VerificationPhase,
    status: VerificationStatus,
    steps: &[VerificationStepReport],
) -> String {
    let mut summary = format!(
        "[verifier:{}:{}] {} ({})",
        phase.as_str(),
        adapter_id,
        status.as_str(),
        project_root.display()
    );
    if steps.is_empty() {
        summary.push_str("\n[verifier] no verification steps were planned");
        return summary;
    }
    let primary = steps
        .iter()
        .find(|step| !step.status.is_success())
        .unwrap_or(&steps[0]);
    let label = match primary.status {
        VerificationStatus::Passed => "ok",
        VerificationStatus::Failed => "FAIL",
        VerificationStatus::Skipped => "skipped",
        VerificationStatus::Unavailable => "unavailable",
    };
    let failure_suffix = primary
        .failure_kind
        .map(|kind| format!(" ({})", kind.as_str()))
        .unwrap_or_default();
    let _ = writeln!(
        summary,
        "\n[verifier] {}: {label}{failure_suffix}",
        primary.label
    );
    if !primary.truncated_output.trim().is_empty() {
        summary.push_str(&primary.truncated_output);
    }
    summary.trim_end().to_string()
}

fn package_manager_run_script(manager: PackageManager, script: &str) -> Vec<String> {
    match manager {
        PackageManager::Npm => vec![
            "npm".to_string(),
            "run".to_string(),
            "--silent".to_string(),
            script.to_string(),
        ],
        PackageManager::Pnpm => vec!["pnpm".to_string(), "run".to_string(), script.to_string()],
        PackageManager::Yarn => vec!["yarn".to_string(), script.to_string()],
        PackageManager::Bun => vec!["bun".to_string(), "run".to_string(), script.to_string()],
    }
}

fn package_manager_exec(manager: PackageManager, binary: &str, args: &[&str]) -> Vec<String> {
    let mut command = match manager {
        PackageManager::Npm => vec![
            "npm".to_string(),
            "exec".to_string(),
            "--".to_string(),
            binary.to_string(),
        ],
        PackageManager::Pnpm => vec!["pnpm".to_string(), "exec".to_string(), binary.to_string()],
        PackageManager::Yarn => vec!["yarn".to_string(), "exec".to_string(), binary.to_string()],
        PackageManager::Bun => vec!["bun".to_string(), "x".to_string(), binary.to_string()],
    };
    command.extend(args.iter().map(ToString::to_string));
    command
}

fn python_module_command(
    runner: &PythonRunner,
    module: &str,
    extra_args: &[&str],
    paths: &[PathBuf],
) -> Vec<String> {
    let mut command = runner.command_prefix.clone();
    command.push(module.to_string());
    command.extend(extra_args.iter().map(ToString::to_string));
    command.extend(paths.iter().map(|path| path.display().to_string()));
    command
}

fn has_script(package_value: &Value, script: &str) -> bool {
    package_value
        .get("scripts")
        .and_then(Value::as_object)
        .is_some_and(|scripts| scripts.get(script).and_then(Value::as_str).is_some())
}

fn detect_package_manager(root: &Path) -> PackageManager {
    if root.join("pnpm-lock.yaml").is_file() {
        PackageManager::Pnpm
    } else if root.join("yarn.lock").is_file() {
        PackageManager::Yarn
    } else if root.join("bun.lockb").is_file() || root.join("bun.lock").is_file() {
        PackageManager::Bun
    } else {
        PackageManager::Npm
    }
}

fn build_python_profile_for_path(path: &Path) -> Option<PythonProjectProfile> {
    let project_root = nearest_python_root(path)?;
    build_python_profile_for_root(&project_root)
}

fn build_python_profile_for_root(root: &Path) -> Option<PythonProjectProfile> {
    let project_root = normalize_local_path(root)?;
    let pyproject_path = project_root.join("pyproject.toml");
    let (pyproject_parsed, pyproject_value, pyproject_parse_error) =
        parse_optional_pyproject(&pyproject_path);
    let (runner, launcher_kind) = detect_python_runner(&project_root);
    let test_root_present = project_root.join("tests").is_dir();
    let has_ruff = python_has_ruff(&project_root, pyproject_value.as_ref());
    let has_mypy = python_has_mypy(&project_root, pyproject_value.as_ref());
    let has_pytest = python_has_pytest(&project_root, pyproject_value.as_ref(), test_root_present);
    let typed_targets = python_typed_targets(&project_root, pyproject_value.as_ref());

    Some(PythonProjectProfile {
        project_root,
        runner,
        launcher_kind,
        pyproject_parsed,
        has_ruff,
        has_mypy,
        has_pytest,
        typed_targets,
        test_root_present,
        pyproject_path: pyproject_path.is_file().then_some(pyproject_path),
        pyproject_parse_error,
    })
}

fn parse_optional_pyproject(path: &Path) -> (bool, Option<TomlValue>, Option<String>) {
    if !path.is_file() {
        return (true, None, None);
    }
    match fs::read_to_string(path) {
        Ok(contents) => match contents.parse::<TomlValue>() {
            Ok(value) => (true, Some(value), None),
            Err(error) => (
                false,
                None,
                Some(format!("failed to parse pyproject.toml: {error}")),
            ),
        },
        Err(error) => (
            false,
            None,
            Some(format!("failed to read pyproject.toml: {error}")),
        ),
    }
}

fn detect_python_runner(root: &Path) -> (PythonRunner, PythonLauncherKind) {
    if root.join("uv.lock").is_file() {
        return (
            PythonRunner {
                command_prefix: vec![
                    "uv".to_string(),
                    "run".to_string(),
                    "python".to_string(),
                    "-m".to_string(),
                ],
            },
            PythonLauncherKind::Uv,
        );
    }
    if root.join("poetry.lock").is_file() {
        return (
            PythonRunner {
                command_prefix: vec![
                    "poetry".to_string(),
                    "run".to_string(),
                    "python".to_string(),
                    "-m".to_string(),
                ],
            },
            PythonLauncherKind::Poetry,
        );
    }
    if let Some(interpreter) = find_venv_python(root) {
        return (
            PythonRunner {
                command_prefix: vec![interpreter.display().to_string(), "-m".to_string()],
            },
            PythonLauncherKind::Venv,
        );
    }
    (
        PythonRunner {
            command_prefix: vec!["python".to_string(), "-m".to_string()],
        },
        PythonLauncherKind::Global,
    )
}

fn find_venv_python(root: &Path) -> Option<PathBuf> {
    let env_names = [".venv", "venv", "env"];
    let candidate_suffixes = if cfg!(windows) {
        vec![
            PathBuf::from("Scripts/python.exe"),
            PathBuf::from("Scripts/python"),
        ]
    } else {
        vec![PathBuf::from("bin/python"), PathBuf::from("bin/python3")]
    };
    for env_name in env_names {
        for suffix in &candidate_suffixes {
            let candidate = root.join(env_name).join(suffix);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn python_has_ruff(root: &Path, pyproject: Option<&TomlValue>) -> bool {
    RUFF_CONFIG_FILES
        .iter()
        .any(|name| root.join(name).is_file())
        || pyproject.is_some_and(|value| toml_contains_path(value, &["tool", "ruff"]))
}

fn python_has_mypy(root: &Path, pyproject: Option<&TomlValue>) -> bool {
    MYPY_CONFIG_FILES
        .iter()
        .any(|name| root.join(name).is_file())
        || pyproject.is_some_and(|value| toml_contains_path(value, &["tool", "mypy"]))
        || file_contains(root.join("setup.cfg"), "[mypy]")
}

fn python_has_pytest(root: &Path, pyproject: Option<&TomlValue>, test_root_present: bool) -> bool {
    PYTEST_CONFIG_FILES
        .iter()
        .any(|name| root.join(name).is_file())
        || root.join("conftest.py").is_file()
        || test_root_present
        || pyproject
            .is_some_and(|value| toml_contains_path(value, &["tool", "pytest", "ini_options"]))
}

fn python_typed_targets(root: &Path, pyproject: Option<&TomlValue>) -> Vec<PathBuf> {
    let Some(pyproject) = pyproject else {
        return Vec::new();
    };
    let mut targets = toml_string_targets(pyproject, &["tool", "mypy", "files"])
        .into_iter()
        .map(|value| {
            let path = PathBuf::from(value);
            if path.is_absolute() {
                path
            } else {
                root.join(path)
            }
        })
        .collect::<Vec<_>>();
    dedupe_paths(&mut targets);
    targets
}

fn derive_mypy_targets_from_touched(root: &Path, touched_paths: &[PathBuf]) -> Vec<PathBuf> {
    let python_files = python_source_targets(root, touched_paths);
    let mut targets = python_files
        .iter()
        .map(|path| package_root_for_python_file(root, path))
        .collect::<Vec<_>>();
    dedupe_paths(&mut targets);
    targets
}

fn python_source_targets(root: &Path, touched_paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut paths = touched_paths
        .iter()
        .filter_map(|path| normalize_project_path(root, path))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("py"))
        })
        .collect::<Vec<_>>();
    dedupe_paths(&mut paths);
    paths
}

fn package_root_for_python_file(root: &Path, path: &Path) -> PathBuf {
    let Some(mut cursor) = path.parent().map(Path::to_path_buf) else {
        return path.to_path_buf();
    };
    let mut package_root = None;
    loop {
        if !cursor.starts_with(root) || !cursor.join("__init__.py").is_file() {
            break;
        }
        package_root = Some(cursor.clone());
        let Some(parent) = cursor.parent() else {
            break;
        };
        cursor = parent.to_path_buf();
    }
    package_root.unwrap_or_else(|| path.to_path_buf())
}

fn dedupe_paths(paths: &mut Vec<PathBuf>) {
    let mut seen = Vec::<PathBuf>::new();
    paths.retain(|path| {
        if seen.iter().any(|existing| existing == path) {
            false
        } else {
            seen.push(path.clone());
            true
        }
    });
}

fn normalize_local_path(path: &Path) -> Option<PathBuf> {
    if path.is_absolute() {
        Some(path.to_path_buf())
    } else {
        std::env::current_dir().ok().map(|cwd| cwd.join(path))
    }
}

fn normalize_project_path(root: &Path, path: &Path) -> Option<PathBuf> {
    let absolute = normalize_local_path(path)?;
    absolute.starts_with(root).then_some(absolute)
}

fn toml_contains_path(value: &TomlValue, path: &[&str]) -> bool {
    toml_value_at(value, path).is_some()
}

fn toml_string_targets(value: &TomlValue, path: &[&str]) -> Vec<String> {
    let Some(value) = toml_value_at(value, path) else {
        return Vec::new();
    };
    if let Some(raw) = value.as_str() {
        return raw
            .split([',', '\n'])
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(ToOwned::to_owned)
            .collect();
    }
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(TomlValue::as_str)
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn toml_value_at<'a>(value: &'a TomlValue, path: &[&str]) -> Option<&'a TomlValue> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

fn file_contains(path: impl AsRef<Path>, needle: &str) -> bool {
    fs::read_to_string(path.as_ref()).is_ok_and(|contents| contents.contains(needle))
}

fn nearest_file(start: &Path, file_name: &str) -> Option<PathBuf> {
    let start = if start.is_absolute() {
        start.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(start)
    };
    let mut cursor = if start.is_dir() {
        start
    } else {
        start.parent()?.to_path_buf()
    };
    loop {
        let candidate = cursor.join(file_name);
        if candidate.is_file() {
            return Some(candidate);
        }
        if !cursor.pop() {
            return None;
        }
    }
}

fn nearest_python_root(start: &Path) -> Option<PathBuf> {
    let start = normalize_local_path(start)?;
    let mut cursor = if start.is_dir() {
        start
    } else {
        start.parent()?.to_path_buf()
    };
    loop {
        for marker in PYTHON_ROOT_MARKERS {
            if cursor.join(marker).is_file() {
                return Some(cursor.clone());
            }
        }
        if !cursor.pop() {
            return None;
        }
    }
}

fn classify_step_failure(step: &PlannedStep, body: &str) -> Option<VerificationFailureKind> {
    match step.diagnostics {
        StepDiagnostics::Generic => None,
        StepDiagnostics::Python {
            launcher_kind,
            step_kind,
        } => Some(classify_python_failure_kind(launcher_kind, step_kind, body)),
    }
}

fn classify_step_timeout(step: &PlannedStep) -> Option<VerificationFailureKind> {
    match step.diagnostics {
        StepDiagnostics::Generic => None,
        StepDiagnostics::Python { .. } => Some(VerificationFailureKind::Timeout),
    }
}

fn classify_step_unavailable(step: &PlannedStep, message: &str) -> Option<VerificationFailureKind> {
    match step.diagnostics {
        StepDiagnostics::Generic => None,
        StepDiagnostics::Python { launcher_kind, .. } => {
            Some(classify_python_unavailable_kind(launcher_kind, message))
        }
    }
}

fn classify_python_failure_kind(
    launcher_kind: PythonLauncherKind,
    step_kind: PythonStepKind,
    body: &str,
) -> VerificationFailureKind {
    let lower = body.to_ascii_lowercase();
    if is_python_config_failure(&lower) {
        return VerificationFailureKind::Config;
    }
    if is_python_tool_unavailable(step_kind, &lower) {
        return VerificationFailureKind::ToolUnavailable;
    }
    if is_python_environment_failure(launcher_kind, &lower) {
        return VerificationFailureKind::Environment;
    }
    VerificationFailureKind::Code
}

fn classify_python_unavailable_kind(
    launcher_kind: PythonLauncherKind,
    message: &str,
) -> VerificationFailureKind {
    let lower = message.to_ascii_lowercase();
    if matches!(
        launcher_kind,
        PythonLauncherKind::Uv | PythonLauncherKind::Poetry
    ) && (lower.contains("not found")
        || lower.contains("cannot find")
        || lower.contains("could not find"))
    {
        VerificationFailureKind::ToolUnavailable
    } else {
        VerificationFailureKind::Environment
    }
}

fn is_python_config_failure(lower: &str) -> bool {
    (lower.contains("pyproject.toml")
        && (lower.contains("parse") || lower.contains("invalid") || lower.contains("config")))
        || lower.contains("invalid configuration")
        || lower.contains("failed to parse")
        || lower.contains("toml parse error")
}

fn is_python_tool_unavailable(step_kind: PythonStepKind, lower: &str) -> bool {
    let module_name = match step_kind {
        PythonStepKind::RuffCheck => "ruff",
        PythonStepKind::Mypy => "mypy",
        PythonStepKind::Pytest => "pytest",
        PythonStepKind::PyCompile => "py_compile",
    };
    lower.contains(&format!("no module named {module_name}"))
        || lower.contains(&format!("no module named '{module_name}'"))
        || lower.contains(&format!("module named {module_name}"))
        || lower.contains(&format!("{module_name} is not installed"))
        || (lower.contains("command not found") && lower.contains(module_name))
        || (lower.contains("not recognized as an internal or external command")
            && lower.contains(module_name))
}

fn is_python_environment_failure(launcher_kind: PythonLauncherKind, lower: &str) -> bool {
    if lower.contains("virtualenv")
        || lower.contains("venv")
        || lower.contains("interpreter")
        || lower.contains("dependency resolution")
        || lower.contains("environment")
        || lower.contains("failed to create")
        || lower.contains("no such file or directory")
        || lower.contains("cannot find the path specified")
        || lower.contains("poetry could not find")
    {
        return true;
    }
    matches!(
        launcher_kind,
        PythonLauncherKind::Venv | PythonLauncherKind::Global
    ) && (lower.contains("python executable") || lower.contains("python was not found"))
}

fn run_step(
    cwd: &Path,
    step: &PlannedStep,
    timeout: Duration,
    max_output_bytes: usize,
) -> StepOutcome {
    let mut command = Command::new(&step.command[0]);
    command.current_dir(cwd);
    command.stdin(std::process::Stdio::null());
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::piped());
    if step.command.first().is_some_and(|bin| {
        let name = std::path::Path::new(bin)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(bin);
        name == "cargo"
    }) {
        command.env("CARGO_TERM_COLOR", "never");
    }
    for arg in step.command.iter().skip(1) {
        command.arg(arg);
    }

    let started = Instant::now();
    match spawn_with_timeout(command, timeout) {
        Ok(output) => {
            let duration_ms = duration_millis_u64(started.elapsed());
            let mut body = String::new();
            body.push_str(&String::from_utf8_lossy(&output.stdout));
            if !output.stderr.is_empty() {
                if !body.is_empty() {
                    body.push('\n');
                }
                body.push_str(&String::from_utf8_lossy(&output.stderr));
            }
            let body = truncate_output(&body, max_output_bytes);
            if output.status.success() {
                StepOutcome::Passed { body, duration_ms }
            } else {
                let failure_kind = classify_step_failure(step, &body);
                StepOutcome::Failed {
                    body,
                    duration_ms,
                    failure_kind,
                }
            }
        }
        Err(SpawnError::Timeout) => {
            let body = truncate_output(
                &format!("step timed out after {}s", timeout.as_secs()),
                max_output_bytes,
            );
            StepOutcome::Failed {
                failure_kind: classify_step_timeout(step),
                body,
                duration_ms: duration_millis_u64(started.elapsed()),
            }
        }
        Err(SpawnError::Io(error)) => {
            let message = truncate_output(&error.to_string(), max_output_bytes);
            StepOutcome::Unavailable {
                failure_kind: classify_step_unavailable(step, &message),
                message,
                duration_ms: duration_millis_u64(started.elapsed()),
            }
        }
    }
}

fn duration_millis_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[derive(Debug)]
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

    let mut child = command.spawn().map_err(SpawnError::Io)?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let (tx, rx) = mpsc::channel();
    let stdout_handle = stdout.map(|mut stream| {
        let tx = tx.clone();
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = std::io::Read::read_to_end(&mut stream, &mut bytes);
            let _ = tx.send(("stdout", bytes));
        })
    });
    let stderr_handle = stderr.map(|mut stream| {
        let tx = tx.clone();
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = std::io::Read::read_to_end(&mut stream, &mut bytes);
            let _ = tx.send(("stderr", bytes));
        })
    });
    drop(tx);

    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().map_err(SpawnError::Io)? {
            if let Some(handle) = stdout_handle {
                let _ = handle.join();
            }
            if let Some(handle) = stderr_handle {
                let _ = handle.join();
            }
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            while let Ok((which, bytes)) = rx.try_recv() {
                if which == "stdout" {
                    stdout = bytes;
                } else {
                    stderr = bytes;
                }
            }
            return Ok(std::process::Output {
                status,
                stdout,
                stderr,
            });
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(SpawnError::Timeout);
        }
        thread::sleep(Duration::from_millis(50));
    }
}

/// Merge a structured report back into the legacy tool-result channel.
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

/// Trim output to the configured byte budget, preserving the head, tail, and
/// diagnostically relevant lines.
#[must_use]
pub fn truncate_output(body: &str, max_bytes: usize) -> String {
    if body.len() <= max_bytes {
        return body.to_string();
    }
    let head_budget = max_bytes / 2;
    let tail_budget = max_bytes / 4;
    let signal_budget = max_bytes.saturating_sub(head_budget + tail_budget + 32);

    let mut head = String::new();
    for line in body.lines() {
        if head.len() + line.len() + 1 > head_budget {
            break;
        }
        head.push_str(line);
        head.push('\n');
    }

    let mut tail = String::new();
    for line in body.lines().rev() {
        if tail.len() + line.len() + 1 > tail_budget {
            break;
        }
        tail = format!("{line}\n{tail}");
    }

    let mut signals = String::new();
    for line in body.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("error")
            || lower.contains("warning")
            || lower.contains("failed")
            || lower.contains("panic")
            || lower.contains("traceback")
        {
            if signals.len() + line.len() + 1 > signal_budget {
                break;
            }
            signals.push_str(line);
            signals.push('\n');
        }
    }

    let mut out = String::new();
    out.push_str(&head);
    out.push_str("... (truncated) ...\n");
    if !signals.trim().is_empty() {
        out.push_str(&signals);
        out.push_str("... (tail) ...\n");
    }
    out.push_str(&tail);
    out.trim_end().to_string()
}

fn extract_file_path(input: &str) -> Option<PathBuf> {
    let value: Value = serde_json::from_str(input).ok()?;
    let path = value
        .get("file_path")
        .or_else(|| value.get("filePath"))
        .or_else(|| value.get("path"))?
        .as_str()?;
    Some(PathBuf::from(path))
}

fn next_report_id() -> String {
    format!("vr-{}", REPORT_COUNTER.fetch_add(1, Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("verifier-unit-{tag}-{nanos}"));
        fs::create_dir_all(&root).expect("temp dir should create");
        root
    }

    #[test]
    fn truncate_output_keeps_signal_lines_and_tail() {
        let mut body = String::new();
        for index in 0..400 {
            let _ = writeln!(body, "noise line {index}");
        }
        body.push_str("warning: some warning\n");
        body.push_str("Traceback (most recent call last):\n");
        body.push_str("panic: boom\n");
        let truncated = truncate_output(&body, 512);
        assert!(truncated.contains("warning: some warning"));
        assert!(truncated.contains("Traceback"));
        assert!(truncated.contains("panic: boom"));
        assert!(truncated.contains("... (truncated) ..."));
    }

    #[test]
    fn prepend_verifier_summary_merges_with_output() {
        let merged = prepend_verifier_summary("[verifier] ok", "edited".to_string());
        assert!(merged.contains("edited"));
        assert!(merged.contains("[verifier output]"));
    }

    #[test]
    fn extract_file_path_supports_known_keys() {
        assert_eq!(
            extract_file_path(r#"{"file_path":"src/lib.rs"}"#),
            Some(PathBuf::from("src/lib.rs"))
        );
        assert_eq!(
            extract_file_path(r#"{"filePath":"src/lib.rs"}"#),
            Some(PathBuf::from("src/lib.rs"))
        );
        assert_eq!(
            extract_file_path(r#"{"path":"src/lib.rs"}"#),
            Some(PathBuf::from("src/lib.rs"))
        );
    }

    #[test]
    fn gate_status_defaults_to_not_required() {
        let gate = VerificationGateStatus::not_required();
        assert!(!gate.attempted);
        assert!(gate.passed);
    }

    #[test]
    fn python_profile_parses_pyproject_and_detects_tools() {
        let root = temp_dir("pyproject");
        fs::write(
            root.join("pyproject.toml"),
            r#"
[tool.ruff]
line-length = 100

[tool.mypy]
files = ["app", "tests"]

[tool.pytest.ini_options]
addopts = "-q"
"#,
        )
        .expect("pyproject should write");

        let profile = build_python_profile_for_root(&root).expect("profile should build");

        assert!(profile.pyproject_parsed);
        assert!(profile.has_ruff);
        assert!(profile.has_mypy);
        assert!(profile.has_pytest);
        assert_eq!(
            profile.typed_targets,
            vec![root.join("app"), root.join("tests")]
        );

        fs::remove_dir_all(root).expect("temp dir should clean up");
    }

    #[test]
    fn python_profile_marks_invalid_pyproject_as_config_failure() {
        let root = temp_dir("bad-pyproject");
        fs::write(root.join("pyproject.toml"), "[tool.ruff\n").expect("pyproject should write");

        let profile = build_python_profile_for_root(&root).expect("profile should build");
        let report = python_config_failure_report(
            &profile,
            VerificationPhase::Quick,
            vec![root.join("pyproject.toml")],
            2_048,
        )
        .expect("invalid pyproject should report failure");

        assert!(!profile.pyproject_parsed);
        assert_eq!(report.status, VerificationStatus::Failed);
        assert_eq!(
            report.steps[0].failure_kind,
            Some(VerificationFailureKind::Config)
        );

        fs::remove_dir_all(root).expect("temp dir should clean up");
    }

    #[test]
    fn detect_python_runner_prefers_uv_over_venv() {
        let root = temp_dir("uv-runner");
        fs::write(root.join("uv.lock"), "").expect("uv lock should write");
        let venv_python = if cfg!(windows) {
            root.join(".venv").join("Scripts").join("python.exe")
        } else {
            root.join(".venv").join("bin").join("python")
        };
        fs::create_dir_all(venv_python.parent().expect("venv parent"))
            .expect("venv dir should create");
        fs::write(&venv_python, "").expect("fake interpreter should write");

        let (runner, launcher_kind) = detect_python_runner(&root);

        assert_eq!(launcher_kind, PythonLauncherKind::Uv);
        assert_eq!(
            runner.command_prefix,
            vec![
                "uv".to_string(),
                "run".to_string(),
                "python".to_string(),
                "-m".to_string()
            ]
        );

        fs::remove_dir_all(root).expect("temp dir should clean up");
    }

    #[test]
    fn detect_python_runner_uses_local_venv_when_present() {
        let root = temp_dir("venv-runner");
        let venv_python = if cfg!(windows) {
            root.join("venv").join("Scripts").join("python.exe")
        } else {
            root.join("venv").join("bin").join("python")
        };
        fs::create_dir_all(venv_python.parent().expect("venv parent"))
            .expect("venv dir should create");
        fs::write(&venv_python, "").expect("fake interpreter should write");

        let (_runner, launcher_kind) = detect_python_runner(&root);

        assert_eq!(launcher_kind, PythonLauncherKind::Venv);

        fs::remove_dir_all(root).expect("temp dir should clean up");
    }

    #[test]
    fn detect_python_runner_uses_poetry_when_lock_present() {
        let root = temp_dir("poetry-runner");
        fs::write(root.join("poetry.lock"), "").expect("poetry lock should write");

        let (runner, launcher_kind) = detect_python_runner(&root);

        assert_eq!(launcher_kind, PythonLauncherKind::Poetry);
        assert_eq!(
            runner.command_prefix,
            vec![
                "poetry".to_string(),
                "run".to_string(),
                "python".to_string(),
                "-m".to_string()
            ]
        );

        fs::remove_dir_all(root).expect("temp dir should clean up");
    }

    #[test]
    fn nearest_python_root_prefers_closest_matching_directory() {
        let root = temp_dir("python-root");
        let nested = root.join("pkg").join("inner");
        fs::create_dir_all(&nested).expect("nested dir should create");
        fs::write(root.join("pyproject.toml"), "[project]\nname = 'root'\n")
            .expect("root pyproject should write");
        fs::write(nested.join("requirements.txt"), "pytest\n")
            .expect("nested requirements should write");

        let detected = nearest_python_root(&nested.join("module.py")).expect("root should resolve");

        assert_eq!(detected, nested);

        fs::remove_dir_all(root).expect("temp dir should clean up");
    }

    #[test]
    fn python_quick_steps_fall_back_to_py_compile() {
        let root = temp_dir("pycompile");
        fs::write(root.join("requirements.txt"), "pytest\n").expect("requirements should write");
        fs::write(root.join("main.py"), "print('ok')\n").expect("python file should write");

        let profile = build_python_profile_for_root(&root).expect("profile should build");
        let steps = python_quick_steps(&profile, &[root.join("main.py")]);

        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].label, "python -m py_compile");

        fs::remove_dir_all(root).expect("temp dir should clean up");
    }

    #[test]
    fn derive_mypy_targets_uses_package_root() {
        let root = temp_dir("mypy-targets");
        let pkg = root.join("pkg");
        let sub = pkg.join("sub");
        fs::create_dir_all(&sub).expect("package dir should create");
        fs::write(pkg.join("__init__.py"), "").expect("init should write");
        fs::write(sub.join("__init__.py"), "").expect("sub init should write");
        fs::write(sub.join("mod.py"), "x = 1\n").expect("module should write");

        let targets = derive_mypy_targets_from_touched(&root, &[sub.join("mod.py")]);

        assert_eq!(targets, vec![pkg]);

        fs::remove_dir_all(root).expect("temp dir should clean up");
    }

    #[test]
    fn classify_python_failures_distinguishes_tool_environment_and_config() {
        assert_eq!(
            classify_python_failure_kind(
                PythonLauncherKind::Global,
                PythonStepKind::RuffCheck,
                "No module named ruff",
            ),
            VerificationFailureKind::ToolUnavailable
        );
        assert_eq!(
            classify_python_failure_kind(
                PythonLauncherKind::Poetry,
                PythonStepKind::Pytest,
                "Poetry could not find a compatible environment",
            ),
            VerificationFailureKind::Environment
        );
        assert_eq!(
            classify_python_failure_kind(
                PythonLauncherKind::Global,
                PythonStepKind::Mypy,
                "failed to parse pyproject.toml",
            ),
            VerificationFailureKind::Config
        );
        assert_eq!(
            classify_step_timeout(&PlannedStep {
                label: "pytest".to_string(),
                command: vec!["python".to_string(), "-m".to_string(), "pytest".to_string()],
                diagnostics: StepDiagnostics::Python {
                    launcher_kind: PythonLauncherKind::Global,
                    step_kind: PythonStepKind::Pytest,
                },
            }),
            Some(VerificationFailureKind::Timeout)
        );
    }
}
