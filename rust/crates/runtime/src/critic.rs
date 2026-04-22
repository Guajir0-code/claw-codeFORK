//! Critic subagent gate.
//!
//! A thin planner that decides whether to invoke a second-opinion "critic"
//! subagent on a mutation. The critic itself is expected to be a cheap-model
//! pass that re-reads a diff and reinjects at most P0/P1 findings back into the
//! main turn. This module only encodes:
//!   * Diff-size thresholds (≥4 files OR ≥200 added/removed lines OR >1 root).
//!   * A `subagent_depth` guard so critic calls never nest.
//!   * A dedup set so each `mutation_sequence` triggers at most one critic run.
//!   * The preferred cheap-model hint.
//!
//! Keeping the policy isolated lets the runtime wire it into `conversation.rs`
//! without that file having to know the threshold numerics.

use std::collections::HashSet;

/// Diff-size snapshot fed to the critic planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffStats {
    pub files_changed: usize,
    pub lines_changed: usize,
    pub distinct_roots: usize,
}

/// Why the critic was (or was not) invoked. Emitted into telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CriticDecision {
    /// Critic should run for `mutation_sequence`. Carries the reason that
    /// tipped the threshold.
    Run { reason: String },
    /// Critic should not run.
    Skip { reason: String },
}

/// Thresholds per spec: ≥4 files OR ≥200 lines OR >1 root.
pub const CRITIC_FILE_THRESHOLD: usize = 4;
pub const CRITIC_LINE_THRESHOLD: usize = 200;
pub const CRITIC_ROOT_THRESHOLD: usize = 1; // strictly more than 1

/// Model hint the runtime should use when spawning the critic subagent.
/// Kept as a free-form string to avoid a compile-time coupling to model IDs.
pub const CRITIC_MODEL_HINT: &str = "claude-haiku";

/// Planner that tracks which mutation sequences have already been audited.
///
/// Callers construct one per conversation and call [`CriticPlanner::plan`] each
/// time a new mutation finalizes. The planner is intentionally infallible —
/// actually spawning the subagent and reinjecting findings stays in the caller.
#[derive(Debug, Default)]
pub struct CriticPlanner {
    audited: HashSet<u64>,
}

impl CriticPlanner {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Decide whether to invoke the critic for this mutation. Records the
    /// sequence as audited only when the decision is [`CriticDecision::Run`] —
    /// skips from depth or thresholds do NOT consume the slot, so a later
    /// mutation with the same sequence number (shouldn't happen, but still)
    /// would be evaluated fresh.
    pub fn plan(
        &mut self,
        mutation_sequence: u64,
        subagent_depth: u32,
        stats: DiffStats,
    ) -> CriticDecision {
        if subagent_depth > 0 {
            return CriticDecision::Skip {
                reason: format!("nested subagent depth={subagent_depth}"),
            };
        }
        if self.audited.contains(&mutation_sequence) {
            return CriticDecision::Skip {
                reason: format!("already audited mutation_sequence={mutation_sequence}"),
            };
        }
        let tripped = trip_reason(stats);
        match tripped {
            Some(reason) => {
                self.audited.insert(mutation_sequence);
                CriticDecision::Run { reason }
            }
            None => CriticDecision::Skip {
                reason: format!(
                    "below thresholds (files={}, lines={}, roots={})",
                    stats.files_changed, stats.lines_changed, stats.distinct_roots
                ),
            },
        }
    }

    #[must_use]
    pub fn has_audited(&self, mutation_sequence: u64) -> bool {
        self.audited.contains(&mutation_sequence)
    }
}

fn trip_reason(stats: DiffStats) -> Option<String> {
    if stats.files_changed >= CRITIC_FILE_THRESHOLD {
        return Some(format!(
            "files_changed={} >= {}",
            stats.files_changed, CRITIC_FILE_THRESHOLD
        ));
    }
    if stats.lines_changed >= CRITIC_LINE_THRESHOLD {
        return Some(format!(
            "lines_changed={} >= {}",
            stats.lines_changed, CRITIC_LINE_THRESHOLD
        ));
    }
    if stats.distinct_roots > CRITIC_ROOT_THRESHOLD {
        return Some(format!(
            "distinct_roots={} > {}",
            stats.distinct_roots, CRITIC_ROOT_THRESHOLD
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> DiffStats {
        DiffStats {
            files_changed: 1,
            lines_changed: 5,
            distinct_roots: 1,
        }
    }

    #[test]
    fn below_thresholds_skips() {
        let mut planner = CriticPlanner::new();
        match planner.plan(1, 0, small()) {
            CriticDecision::Skip { reason } => assert!(reason.starts_with("below thresholds")),
            CriticDecision::Run { .. } => panic!("expected skip, got run"),
        }
    }

    #[test]
    fn file_threshold_runs() {
        let mut planner = CriticPlanner::new();
        let stats = DiffStats {
            files_changed: 4,
            lines_changed: 20,
            distinct_roots: 1,
        };
        assert!(matches!(
            planner.plan(1, 0, stats),
            CriticDecision::Run { .. }
        ));
    }

    #[test]
    fn line_threshold_runs() {
        let mut planner = CriticPlanner::new();
        let stats = DiffStats {
            files_changed: 1,
            lines_changed: 200,
            distinct_roots: 1,
        };
        assert!(matches!(
            planner.plan(1, 0, stats),
            CriticDecision::Run { .. }
        ));
    }

    #[test]
    fn root_threshold_runs_when_strictly_more_than_one() {
        let mut planner = CriticPlanner::new();
        let stats = DiffStats {
            files_changed: 1,
            lines_changed: 5,
            distinct_roots: 2,
        };
        assert!(matches!(
            planner.plan(1, 0, stats),
            CriticDecision::Run { .. }
        ));
    }

    #[test]
    fn single_root_does_not_trip_root_threshold() {
        let mut planner = CriticPlanner::new();
        let stats = DiffStats {
            files_changed: 1,
            lines_changed: 5,
            distinct_roots: 1,
        };
        assert!(matches!(
            planner.plan(1, 0, stats),
            CriticDecision::Skip { .. }
        ));
    }

    #[test]
    fn nested_subagent_depth_blocks_run() {
        let mut planner = CriticPlanner::new();
        let stats = DiffStats {
            files_changed: 10,
            lines_changed: 500,
            distinct_roots: 5,
        };
        let decision = planner.plan(1, 1, stats);
        match decision {
            CriticDecision::Skip { reason } => assert!(reason.contains("nested subagent depth")),
            CriticDecision::Run { .. } => panic!("expected skip, got run"),
        }
        assert!(
            !planner.has_audited(1),
            "depth-blocked run must not consume the mutation slot"
        );
    }

    #[test]
    fn one_run_per_mutation_sequence() {
        let mut planner = CriticPlanner::new();
        let stats = DiffStats {
            files_changed: 4,
            lines_changed: 20,
            distinct_roots: 1,
        };
        assert!(matches!(
            planner.plan(42, 0, stats),
            CriticDecision::Run { .. }
        ));
        match planner.plan(42, 0, stats) {
            CriticDecision::Skip { reason } => assert!(reason.starts_with("already audited")),
            CriticDecision::Run { .. } => panic!("expected skip on dup, got run"),
        }
    }

    #[test]
    fn distinct_mutation_sequences_are_independent() {
        let mut planner = CriticPlanner::new();
        let stats = DiffStats {
            files_changed: 4,
            lines_changed: 20,
            distinct_roots: 1,
        };
        assert!(matches!(
            planner.plan(1, 0, stats),
            CriticDecision::Run { .. }
        ));
        assert!(matches!(
            planner.plan(2, 0, stats),
            CriticDecision::Run { .. }
        ));
    }

    #[test]
    fn model_hint_is_cheap() {
        assert_eq!(CRITIC_MODEL_HINT, "claude-haiku");
    }
}
