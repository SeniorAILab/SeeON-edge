//! Live bed-exit stage joining matched-pose geometry to the existing policy owner.

use super::bed_exit_input::{BedInputError, BedInputProducer};
use super::ingest::{Frame, ObservedPose};
use seeon_worker::bed_exit::{BedExitClocks, BedExitError, BedExitMonitor, BedExitScoring};
use seeon_worker::episode::BusinessEvent;
use seeon_worker::trace::DecisionTraceSnapshot;

/// A borrowed, synchronous decision receipt. The observer is receipt-only and
/// MUST NOT reenter this stage. A fatal partial prefix is already observed by
/// the callback; callers MUST NOT redeliver the prefix from the returned error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BedDecisionUpdate<'a> {
    pub frame: seeon_deepstream_native::FrameIdentity,
    pub frame_index: i64,
    pub time_sec: Option<f64>,
    pub snapshots: &'a [DecisionTraceSnapshot],
    pub events: &'a [BusinessEvent],
    pub scoring: Option<BedExitScoring>,
    /// False only when a fatal policy failure retained a non-empty event prefix.
    pub completed: bool,
}

/// Failure from input production or the real bed-exit policy owner.
#[derive(Clone, Debug, PartialEq)]
pub enum BedStageError {
    Input(BedInputError),
    Policy(BedExitError),
}

impl std::fmt::Display for BedStageError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input(error) => write!(formatter, "bed stage input failure: {error}"),
            Self::Policy(error) => write!(formatter, "bed stage policy failure: {error}"),
        }
    }
}

impl std::error::Error for BedStageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Input(error) => Some(error),
            Self::Policy(error) => Some(error),
        }
    }
}

/// Per-camera owner of live input production and bed-exit decision state.
pub struct BedStage {
    monitor: BedExitMonitor,
    inputs: BedInputProducer,
}

impl BedStage {
    pub fn new(monitor: BedExitMonitor, inputs: BedInputProducer) -> Self {
        Self { monitor, inputs }
    }

    pub fn monitor(&self) -> &BedExitMonitor {
        &self.monitor
    }

    /// Build numeric input from this frame, then update the same monitor once.
    /// Success always reports one receipt, including empty-region updates.
    /// Fatal errors report their exact accepted event prefix before returning;
    /// that receipt is incomplete and the returned prefix MUST NOT be redelivered.
    pub fn observe(
        &mut self,
        frame: &Frame,
        observed: &[ObservedPose],
        clocks: BedExitClocks,
        observer: &mut dyn FnMut(BedDecisionUpdate<'_>),
    ) -> Result<(), BedStageError> {
        let input = self
            .inputs
            .build(frame, observed, self.monitor.assignments())
            .map_err(BedStageError::Input)?;
        match self.monitor.update(&input, clocks) {
            Ok(outcome) => {
                observer(BedDecisionUpdate {
                    frame: frame.identity,
                    frame_index: input.frame_index,
                    time_sec: input.time_sec,
                    snapshots: self.monitor.last_trace_snapshots(),
                    events: &outcome.events,
                    scoring: outcome.scoring_observation,
                    completed: true,
                });
                Ok(())
            }
            Err(error) => {
                if let BedExitError::FatalPartialState { emitted_events, .. } = &error {
                    if !emitted_events.is_empty() {
                        observer(BedDecisionUpdate {
                            frame: frame.identity,
                            frame_index: input.frame_index,
                            time_sec: input.time_sec,
                            snapshots: self.monitor.last_trace_snapshots(),
                            events: emitted_events,
                            scoring: None,
                            completed: false,
                        });
                    }
                }
                Err(BedStageError::Policy(error))
            }
        }
    }

    /// Apply the monitor's coast update without inventing a frame receipt.
    pub fn coast(
        &mut self,
        frame_index: Option<i64>,
        snapshot_at: f64,
    ) -> Result<(), BedStageError> {
        self.monitor
            .coast(frame_index, snapshot_at)
            .map(|_| ())
            .map_err(BedStageError::Policy)
    }
}
