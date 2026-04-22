//! Rollout metrics aggregator and budget gates for the edit→verify→fix loop.
//!
//! The runtime emits individual `verifier_ran` traces per step, plus per-turn
//! token/latency counts. This module collapses those samples into the handful
//! of rollout metrics named by the plan:
//!
//! * `quick_verify_latency_ms` — p50/p95 latency of quick-phase verification.
//! * `final_gate_pass_rate` — fraction of final-phase reports that succeeded.
//! * `repair_iterations_until_green` — mean repair iterations needed before
//!   the first green final-gate report.
//! * `tokens_per_successful_fix` — total tokens spent divided by number of
//!   successful fixes (green final reports).
//! * `turn_latency_ms` — mean end-to-end turn latency.
//!
//! It also codifies the rollout **budget gates** from the plan:
//! * pass-rate may not regress by more than 1 percentage point.
//! * repair-iteration mean may not regress by more than 5%.
//! * tokens-per-fix may not regress by more than 10%.
//! * turn-latency p50 may not regress by more than 15%.
//!
//! Keeping aggregation + gates in one crate-visible module lets CI wire it
//! without reaching into `conversation.rs` internals.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::map_unwrap_or
)]

use std::cmp::Ordering;

use telemetry::SessionTraceRecord;

/// Phase observed for a single verifier step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifierPhase {
    Quick,
    Final,
}

/// A single verifier-step observation. The aggregator takes a slice of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifierSample {
    pub phase: VerifierPhase,
    /// Identifies the repair episode (0-based index of a user turn, typically).
    pub turn_index: u64,
    /// Monotonic within a turn; the final successful report carries the
    /// "final iteration count" for the episode.
    pub iteration: u32,
    pub duration_ms: u64,
    pub succeeded: bool,
}

/// Per-turn fact used for token / latency rollups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnSample {
    pub turn_index: u64,
    pub tokens_total: u64,
    pub turn_latency_ms: u64,
    /// True iff the turn ended with a green final-gate report.
    pub successful_fix: bool,
}

/// Aggregated rollout metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RolloutMetrics {
    pub quick_verify_latency_p50_ms: f64,
    pub quick_verify_latency_p95_ms: f64,
    pub final_gate_pass_rate: f64,
    pub repair_iterations_until_green_mean: f64,
    pub tokens_per_successful_fix: f64,
    pub turn_latency_p50_ms: f64,
    pub turn_latency_mean_ms: f64,
    pub samples: AggregateCounts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AggregateCounts {
    pub quick_samples: usize,
    pub final_samples: usize,
    pub turns: usize,
    pub successful_fixes: usize,
}

/// Collapse raw samples into [`RolloutMetrics`]. Missing signals become `0.0`
/// (not NaN) so downstream gates can compare safely.
#[must_use]
pub fn aggregate(verifier: &[VerifierSample], turns: &[TurnSample]) -> RolloutMetrics {
    let mut quick_latencies: Vec<u64> = verifier
        .iter()
        .filter(|s| s.phase == VerifierPhase::Quick)
        .map(|s| s.duration_ms)
        .collect();
    let final_samples: Vec<&VerifierSample> = verifier
        .iter()
        .filter(|s| s.phase == VerifierPhase::Final)
        .collect();
    let final_success_count = final_samples.iter().filter(|s| s.succeeded).count();

    // Repair iterations per turn: count of final-phase failures before the
    // first green final-phase report (if any). If the turn never went green,
    // treat the whole chain as the cost.
    let mut repair_iterations: Vec<u32> = Vec::new();
    for turn in turns {
        let turn_finals: Vec<&&VerifierSample> = final_samples
            .iter()
            .filter(|s| s.turn_index == turn.turn_index)
            .collect();
        if turn_finals.is_empty() {
            continue;
        }
        let mut failures = 0_u32;
        let mut saw_green = false;
        for sample in &turn_finals {
            if sample.succeeded {
                saw_green = true;
                break;
            }
            failures += 1;
        }
        if saw_green || !turn_finals.is_empty() {
            repair_iterations.push(failures);
        }
    }

    let tokens_total: u64 = turns.iter().map(|t| t.tokens_total).sum();
    let successful_fixes = turns.iter().filter(|t| t.successful_fix).count();
    let tokens_per_successful_fix = if successful_fixes == 0 {
        0.0
    } else {
        tokens_total as f64 / successful_fixes as f64
    };

    let turn_latencies: Vec<u64> = turns.iter().map(|t| t.turn_latency_ms).collect();

    RolloutMetrics {
        quick_verify_latency_p50_ms: percentile(&mut quick_latencies.clone(), 0.50),
        quick_verify_latency_p95_ms: percentile(&mut quick_latencies, 0.95),
        final_gate_pass_rate: if final_samples.is_empty() {
            0.0
        } else {
            final_success_count as f64 / final_samples.len() as f64
        },
        repair_iterations_until_green_mean: mean_u32(&repair_iterations),
        tokens_per_successful_fix,
        turn_latency_p50_ms: percentile(&mut turn_latencies.clone(), 0.50),
        turn_latency_mean_ms: mean_u64(&turn_latencies),
        samples: AggregateCounts {
            quick_samples: verifier
                .iter()
                .filter(|s| s.phase == VerifierPhase::Quick)
                .count(),
            final_samples: final_samples.len(),
            turns: turns.len(),
            successful_fixes,
        },
    }
}

fn percentile(values: &mut [u64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable();
    let rank = (p * (values.len() as f64 - 1.0)).round() as usize;
    values[rank.min(values.len() - 1)] as f64
}

fn mean_u32(values: &[u32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|v| f64::from(*v)).sum::<f64>() / values.len() as f64
}

fn mean_u64(values: &[u64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64
}

/// Extract `(VerifierSample, TurnSample)` vectors from raw session trace records.
///
/// Recognises:
/// * `verifier_ran` records with attributes `phase` ("quick" | "final"),
///   `mutation_sequence`, `duration_ms`, `passed`. `mutation_sequence`
///   doubles as the `turn_index` so samples from the same repair episode
///   group together even if the caller did not set an explicit turn id.
/// * `turn_completed` records with optional `tokens_total`, `turn_latency_ms`,
///   `verification_gate_passed`. Records missing those fields are still
///   admitted with zeros so count-based aggregates stay honest.
///
/// Any trace with an unexpected name is ignored.
#[must_use]
pub fn samples_from_traces(
    traces: &[SessionTraceRecord],
) -> (Vec<VerifierSample>, Vec<TurnSample>) {
    let mut verifier = Vec::new();
    let mut turns = Vec::new();
    let mut turn_counter: u64 = 0;

    for record in traces {
        match record.name.as_str() {
            "verifier_ran" => {
                let phase = record
                    .attributes
                    .get("phase")
                    .and_then(|v| v.as_str())
                    .map(|s| match s {
                        "final" => VerifierPhase::Final,
                        _ => VerifierPhase::Quick,
                    })
                    .unwrap_or(VerifierPhase::Quick);
                let turn_index = record
                    .attributes
                    .get("mutation_sequence")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let iteration = record
                    .attributes
                    .get("iteration")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0);
                let duration_ms = record
                    .attributes
                    .get("duration_ms")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let succeeded = record
                    .attributes
                    .get("passed")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                verifier.push(VerifierSample {
                    phase,
                    turn_index,
                    iteration,
                    duration_ms,
                    succeeded,
                });
            }
            "turn_completed" => {
                let tokens_total = record
                    .attributes
                    .get("tokens_total")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let turn_latency_ms = record
                    .attributes
                    .get("turn_latency_ms")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let successful_fix = record
                    .attributes
                    .get("verification_gate_passed")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                turns.push(TurnSample {
                    turn_index: turn_counter,
                    tokens_total,
                    turn_latency_ms,
                    successful_fix,
                });
                turn_counter += 1;
            }
            _ => {}
        }
    }

    (verifier, turns)
}

/// Budget gate thresholds codified from the rollout plan.
pub const MAX_PASS_RATE_REGRESSION_PP: f64 = 0.01; // 1 percentage point
pub const MAX_REPAIR_ITERATIONS_REGRESSION: f64 = 0.05; // 5% relative
pub const MAX_TOKENS_PER_FIX_REGRESSION: f64 = 0.10; // 10% relative
pub const MAX_TURN_LATENCY_P50_REGRESSION: f64 = 0.15; // 15% relative

/// A single budget-gate violation.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetViolation {
    pub metric: &'static str,
    pub baseline: f64,
    pub current: f64,
    pub limit: f64,
    pub actual: f64,
}

/// Compare a rollout candidate against a baseline. Returns every violation so
/// reports can surface them all at once.
#[must_use]
pub fn evaluate_budget_gates(
    baseline: &RolloutMetrics,
    current: &RolloutMetrics,
) -> Vec<BudgetViolation> {
    let mut violations = Vec::new();

    // Pass-rate: absolute percentage-point drop.
    let pass_rate_drop = baseline.final_gate_pass_rate - current.final_gate_pass_rate;
    if pass_rate_drop > MAX_PASS_RATE_REGRESSION_PP {
        violations.push(BudgetViolation {
            metric: "final_gate_pass_rate",
            baseline: baseline.final_gate_pass_rate,
            current: current.final_gate_pass_rate,
            limit: MAX_PASS_RATE_REGRESSION_PP,
            actual: pass_rate_drop,
        });
    }

    push_relative_regression(
        &mut violations,
        "repair_iterations_until_green_mean",
        baseline.repair_iterations_until_green_mean,
        current.repair_iterations_until_green_mean,
        MAX_REPAIR_ITERATIONS_REGRESSION,
    );
    push_relative_regression(
        &mut violations,
        "tokens_per_successful_fix",
        baseline.tokens_per_successful_fix,
        current.tokens_per_successful_fix,
        MAX_TOKENS_PER_FIX_REGRESSION,
    );
    push_relative_regression(
        &mut violations,
        "turn_latency_p50_ms",
        baseline.turn_latency_p50_ms,
        current.turn_latency_p50_ms,
        MAX_TURN_LATENCY_P50_REGRESSION,
    );

    violations
}

fn push_relative_regression(
    violations: &mut Vec<BudgetViolation>,
    metric: &'static str,
    baseline: f64,
    current: f64,
    limit: f64,
) {
    if baseline <= 0.0 {
        // No baseline signal — cannot compute a relative delta.
        return;
    }
    let delta = (current - baseline) / baseline;
    if delta.partial_cmp(&limit) == Some(Ordering::Greater) {
        violations.push(BudgetViolation {
            metric,
            baseline,
            current,
            limit,
            actual: delta,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vs(phase: VerifierPhase, turn: u64, iter: u32, duration: u64, ok: bool) -> VerifierSample {
        VerifierSample {
            phase,
            turn_index: turn,
            iteration: iter,
            duration_ms: duration,
            succeeded: ok,
        }
    }

    fn ts(turn: u64, tokens: u64, latency: u64, ok: bool) -> TurnSample {
        TurnSample {
            turn_index: turn,
            tokens_total: tokens,
            turn_latency_ms: latency,
            successful_fix: ok,
        }
    }

    #[test]
    fn empty_inputs_produce_zeroed_metrics() {
        let m = aggregate(&[], &[]);
        assert!(m.quick_verify_latency_p50_ms.abs() < f64::EPSILON);
        assert!(m.final_gate_pass_rate.abs() < f64::EPSILON);
        assert!(m.tokens_per_successful_fix.abs() < f64::EPSILON);
        assert_eq!(m.samples.turns, 0);
    }

    #[test]
    fn aggregates_basic_signals() {
        let verifier = vec![
            vs(VerifierPhase::Quick, 0, 1, 10, true),
            vs(VerifierPhase::Quick, 0, 2, 30, true),
            vs(VerifierPhase::Final, 0, 1, 100, false),
            vs(VerifierPhase::Final, 0, 2, 120, true),
            vs(VerifierPhase::Final, 1, 1, 150, true),
        ];
        let turns = vec![ts(0, 1_000, 500, true), ts(1, 500, 400, true)];
        let m = aggregate(&verifier, &turns);
        assert!((m.final_gate_pass_rate - 2.0 / 3.0).abs() < 1e-9);
        assert!((m.repair_iterations_until_green_mean - 0.5).abs() < 1e-9);
        assert!((m.tokens_per_successful_fix - 750.0).abs() < 1e-9);
        assert_eq!(m.samples.successful_fixes, 2);
    }

    #[test]
    fn pass_rate_regression_over_1pp_is_a_violation() {
        let baseline = baseline();
        let mut current = baseline;
        current.final_gate_pass_rate = 0.89; // dropped 2pp from 0.91
        let violations = evaluate_budget_gates(&baseline, &current);
        assert!(violations
            .iter()
            .any(|v| v.metric == "final_gate_pass_rate"));
    }

    #[test]
    fn pass_rate_regression_under_1pp_is_ok() {
        let baseline = baseline();
        let mut current = baseline;
        current.final_gate_pass_rate = 0.905; // dropped 0.5pp
        let violations = evaluate_budget_gates(&baseline, &current);
        assert!(violations.is_empty());
    }

    #[test]
    fn repair_regression_over_5pct_is_a_violation() {
        let baseline = baseline();
        let mut current = baseline;
        current.repair_iterations_until_green_mean = 2.2; // 10% up from 2.0
        let violations = evaluate_budget_gates(&baseline, &current);
        assert!(violations
            .iter()
            .any(|v| v.metric == "repair_iterations_until_green_mean"));
    }

    #[test]
    fn tokens_regression_over_10pct_is_a_violation() {
        let baseline = baseline();
        let mut current = baseline;
        current.tokens_per_successful_fix = 1_200.0; // 20% up from 1000
        let violations = evaluate_budget_gates(&baseline, &current);
        assert!(violations
            .iter()
            .any(|v| v.metric == "tokens_per_successful_fix"));
    }

    #[test]
    fn turn_latency_regression_over_15pct_is_a_violation() {
        let baseline = baseline();
        let mut current = baseline;
        current.turn_latency_p50_ms = 1_200.0; // 20% up from 1000
        let violations = evaluate_budget_gates(&baseline, &current);
        assert!(violations.iter().any(|v| v.metric == "turn_latency_p50_ms"));
    }

    #[test]
    fn flat_metrics_emit_no_violations() {
        let baseline = baseline();
        let current = baseline;
        assert!(evaluate_budget_gates(&baseline, &current).is_empty());
    }

    fn verifier_trace(
        name: &str,
        phase: &str,
        mutation_sequence: u64,
        duration_ms: u64,
        passed: bool,
    ) -> SessionTraceRecord {
        let mut attrs = serde_json::Map::new();
        attrs.insert("phase".into(), serde_json::Value::String(phase.into()));
        attrs.insert(
            "mutation_sequence".into(),
            serde_json::Value::from(mutation_sequence),
        );
        attrs.insert("duration_ms".into(), serde_json::Value::from(duration_ms));
        attrs.insert("passed".into(), serde_json::Value::Bool(passed));
        SessionTraceRecord {
            session_id: "sess".into(),
            sequence: 0,
            name: name.into(),
            timestamp_ms: 0,
            attributes: attrs,
        }
    }

    fn turn_trace(tokens: u64, latency: u64, passed: bool) -> SessionTraceRecord {
        let mut attrs = serde_json::Map::new();
        attrs.insert("tokens_total".into(), serde_json::Value::from(tokens));
        attrs.insert("turn_latency_ms".into(), serde_json::Value::from(latency));
        attrs.insert(
            "verification_gate_passed".into(),
            serde_json::Value::Bool(passed),
        );
        SessionTraceRecord {
            session_id: "sess".into(),
            sequence: 0,
            name: "turn_completed".into(),
            timestamp_ms: 0,
            attributes: attrs,
        }
    }

    #[test]
    fn samples_from_traces_extracts_verifier_and_turn_records() {
        let traces = vec![
            verifier_trace("verifier_ran", "quick", 0, 50, true),
            verifier_trace("verifier_ran", "final", 0, 300, false),
            verifier_trace("verifier_ran", "final", 0, 320, true),
            turn_trace(1_000, 500, true),
            verifier_trace("verifier_ran", "final", 1, 180, true),
            turn_trace(500, 400, true),
            verifier_trace("unrelated_event", "quick", 0, 0, true),
        ];
        let (verifier, turns) = samples_from_traces(&traces);
        assert_eq!(verifier.len(), 4, "unrelated events ignored");
        assert_eq!(turns.len(), 2);
        let metrics = aggregate(&verifier, &turns);
        assert_eq!(metrics.samples.final_samples, 3);
        assert!((metrics.tokens_per_successful_fix - 750.0).abs() < 1e-9);
    }

    #[test]
    fn samples_from_traces_handles_missing_attributes() {
        let record = SessionTraceRecord {
            session_id: "sess".into(),
            sequence: 0,
            name: "verifier_ran".into(),
            timestamp_ms: 0,
            attributes: serde_json::Map::new(),
        };
        let (verifier, turns) = samples_from_traces(&[record]);
        assert_eq!(verifier.len(), 1);
        assert_eq!(turns.len(), 0);
        assert_eq!(verifier[0].duration_ms, 0);
        assert!(!verifier[0].succeeded);
    }

    fn baseline() -> RolloutMetrics {
        RolloutMetrics {
            quick_verify_latency_p50_ms: 100.0,
            quick_verify_latency_p95_ms: 400.0,
            final_gate_pass_rate: 0.91,
            repair_iterations_until_green_mean: 2.0,
            tokens_per_successful_fix: 1_000.0,
            turn_latency_p50_ms: 1_000.0,
            turn_latency_mean_ms: 1_100.0,
            samples: AggregateCounts {
                quick_samples: 10,
                final_samples: 10,
                turns: 10,
                successful_fixes: 9,
            },
        }
    }
}
