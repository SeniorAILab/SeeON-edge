use std::sync::Arc;
use std::sync::mpsc;

use seeon_deepstream_native::FrameIdentity;
use seeon_ml_worker::msg::FallScore;
use seeon_ml_worker::policy::bed_exit::BedDecisionUpdate;
use seeon_ml_worker::policy::fall::DecisionUpdate;
use seeon_ml_worker::run::pump::{PolicyPump, PolicySink, PumpError};
use seeon_worker::episode::BusinessEvent;
use seeon_worker::trace::DecisionTraceReason;

#[path = "support/bed_pump_fixture.rs"]
mod fixture;
use fixture::{NoWallClock, camera, packet};

#[derive(Default)]
struct Sink {
    bed: Vec<BedReceipt>,
    fall: usize,
}

struct BedReceipt {
    frame: FrameIdentity,
    events: Vec<BusinessEvent>,
    reasons: Vec<DecisionTraceReason>,
}

impl PolicySink for Sink {
    fn preflight_fall_decisions(&mut self, batches: usize) -> Result<(), PumpError> {
        assert!((1..=2).contains(&batches));
        Ok(())
    }

    fn preflight_bed_decision(&mut self) -> Result<(), PumpError> {
        self.bed.try_reserve(1).expect("test bed receipt capacity");
        Ok(())
    }

    fn bed_decision(&mut self, update: BedDecisionUpdate<'_>) -> Result<(), PumpError> {
        assert!(update.completed);
        self.bed.push(BedReceipt {
            frame: update.frame,
            events: update.events.to_vec(),
            reasons: update.snapshots.iter().map(|s| s.reason).collect(),
        });
        Ok(())
    }
    fn decision(&mut self, _: DecisionUpdate<'_>) -> Result<(), PumpError> {
        self.fall += 1;
        Ok(())
    }
    fn score(&mut self, _: FrameIdentity, _: u64, _: &FallScore) -> Result<(), PumpError> {
        panic!("no score responses were supplied")
    }
    fn frame_failure(&mut self, _: FrameIdentity, error: &PumpError) -> Result<(), PumpError> {
        panic!("unexpected frame-local failure: {error:?}")
    }
}

#[test]
fn bed_only_uses_real_matched_poses_without_fall_channels_or_wall_clock() {
    let mut pump = PolicyPump::new(vec![camera(false)], None, Arc::new(NoWallClock)).unwrap();
    let mut sink = Sink::default();
    for sequence in 0..=7 {
        let input = packet(sequence, sequence >= 6, false);
        pump.observe(&input, None, &mut sink).unwrap();
        assert_eq!(sink.bed.last().unwrap().frame, input.frame);
    }
    assert_eq!(sink.bed.len(), 8);
    assert!(
        sink.bed[..7]
            .iter()
            .all(|receipt| receipt.events.is_empty())
    );
    let events = &sink.bed[7].events;
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].person_id, Some(7));
    assert_eq!(events[0].time_sec, 7.0);
    assert_eq!(
        sink.bed[7].reasons,
        vec![DecisionTraceReason::OutsideDwellExit]
    );
    assert_eq!(sink.fall, 0);
    assert_eq!(pump.pending_scores(), 0);
    pump.flush(&mut sink).unwrap();
    assert_eq!(sink.bed.len(), 8);
}

#[test]
fn missing_pose_does_not_turn_track_presence_into_exit_evidence() {
    let mut pump = PolicyPump::new(vec![camera(false)], None, Arc::new(NoWallClock)).unwrap();
    let mut sink = Sink::default();
    for sequence in 0..=7 {
        pump.observe(
            &packet(sequence, sequence >= 6, sequence == 7),
            None,
            &mut sink,
        )
        .unwrap();
    }
    assert!(sink.bed.iter().all(|receipt| receipt.events.is_empty()));
    assert_eq!(
        sink.bed[7].reasons,
        vec![DecisionTraceReason::StaleTrackClear]
    );
}

#[test]
fn mixed_domains_keep_bed_dispatch_independent_of_fall_scores() {
    let mut pump = PolicyPump::new(vec![camera(true)], None, Arc::new(NoWallClock)).unwrap();
    let (sender, receiver) = mpsc::sync_channel(64);
    let mut sink = Sink::default();
    for sequence in 0..=7 {
        pump.observe(
            &packet(sequence, sequence >= 6, false),
            Some(&sender),
            &mut sink,
        )
        .unwrap();
        for _ in receiver.try_iter() {}
    }
    assert_eq!(sink.bed.len(), 8);
    assert_eq!(sink.bed[7].events.len(), 1);
    assert!(sink.fall > 0);
}

#[test]
fn fall_gap_refusal_keeps_the_already_observed_bed_receipt() {
    let mut pump = PolicyPump::new(vec![camera(true)], None, Arc::new(NoWallClock)).unwrap();
    let (sender, _receiver) = mpsc::sync_channel(64);
    let mut sink = Sink::default();
    pump.observe(&packet(0, false, false), Some(&sender), &mut sink)
        .unwrap();
    let next = packet(65, false, false);
    let error = pump.observe(&next, Some(&sender), &mut sink).unwrap_err();
    assert!(matches!(
        error,
        PumpError::Response(seeon_ml_worker::run::policy::FallResponseError::Policy(
            seeon_ml_worker::policy::fall::FallStageError::Gap(_)
        ))
    ));
    assert_eq!(sink.bed.len(), 2);
    assert_eq!(sink.bed[1].frame, next.frame);
}
