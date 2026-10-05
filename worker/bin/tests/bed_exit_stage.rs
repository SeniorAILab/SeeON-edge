use seeon_ml_worker::policy::bed_exit::BedStageError;
use seeon_ml_worker::policy::bed_exit_input::BedInputError;
use seeon_worker::bed_exit::{BedExitClocks, BedExitError, BedExitScoring};
use seeon_worker::trace::DecisionTraceReason;

#[path = "support/bed_stage_fixture.rs"]
mod fixture;
use fixture::{arm_one, inside, outside, pose, room, stage, submit};

#[test]
fn persisted_polygon_pose_arms_then_emits_after_continuous_outside_dwell() {
    let mut stage = stage(4, vec![room()]);
    arm_one(&mut stage, &[7]);
    assert!(stage.monitor().assignments()[&7].armed);
    fixture::assert_retained_box(&stage, 7, [2, 2, 6, 8]);

    let first_outside = submit(&mut stage, 4.0, 5, vec![7], vec![outside(7)]).unwrap();
    assert!(first_outside.events.is_empty());
    assert_eq!(stage.monitor().assignments()[&7].outside_dwell_sec, 1.0);
    let exited = submit(&mut stage, 5.0, 6, vec![7], vec![outside(7)]).unwrap();
    assert_eq!(exited.events.len(), 1);
    assert_eq!(exited.events[0].person_id, Some(7));
    assert_eq!(exited.events[0].time_sec, 5.0);
    assert_eq!(
        exited.snapshots[0].reason,
        DecisionTraceReason::OutsideDwellExit
    );
    assert!(exited.scoring.is_some());
}

#[test]
fn missing_pose_does_not_advance_outside_dwell() {
    let mut stage = stage(4, vec![room()]);
    arm_one(&mut stage, &[7]);
    assert!(
        submit(&mut stage, 4.0, 5, vec![7], vec![outside(7)])
            .unwrap()
            .events
            .is_empty()
    );
    assert_eq!(stage.monitor().assignments()[&7].outside_dwell_sec, 1.0);

    let missing = submit(&mut stage, 5.0, 6, vec![7], vec![]).unwrap();
    assert!(missing.events.is_empty());
    assert_eq!(
        missing.snapshots[0].reason,
        DecisionTraceReason::StaleTrackClear
    );
    assert!(!stage.monitor().assignments().contains_key(&7));
}

#[test]
fn empty_regions_report_the_real_unavailable_trace_without_events() {
    let mut stage = stage(4, vec![]);
    let receipt = submit(&mut stage, 1.0, 12, vec![7], vec![inside(7)]).unwrap();
    assert!(receipt.events.is_empty());
    assert_eq!(receipt.scoring, None);
    assert_eq!(receipt.snapshots.len(), 1);
    assert_eq!(
        receipt.snapshots[0].reason,
        DecisionTraceReason::BedRegionUnavailable
    );
}

#[test]
fn input_refusal_skips_callback_and_leaves_monitor_unchanged() {
    let mut stage = stage(4, vec![room()]);
    let frame = fixture::frame_at(1.0, 13, vec![7]);
    let before_assignments = stage.monitor().assignments().clone();
    let before_traces = stage.monitor().last_trace_snapshots().to_vec();
    let mut callbacks = 0;
    let error = stage
        .observe(
            &frame,
            &[pose(7, [2.5, 2.0, 6.0, 8.0])],
            BedExitClocks {
                wall_time: None,
                observed_at: 1.0,
                snapshot_at: 1.0,
            },
            &mut |_| callbacks += 1,
        )
        .unwrap_err();
    assert_eq!(
        error,
        BedStageError::Input(BedInputError::Conversion("person_bbox"))
    );
    assert_eq!(callbacks, 0);
    assert_eq!(stage.monitor().assignments(), &before_assignments);
    assert_eq!(stage.monitor().last_trace_snapshots(), before_traces);
    assert_eq!(stage.monitor().scoring(), BedExitScoring::default());
}

#[test]
fn coast_uses_monitor_retention_without_a_synthetic_receipt() {
    let mut stage = stage(4, vec![room()]);
    let latest = submit(&mut stage, 0.0, 20, vec![7], vec![inside(7)]).unwrap();
    let assignments = stage.monitor().assignments().clone();
    assert_eq!(stage.coast(Some(88), 0.5), Ok(()));
    assert_eq!(stage.monitor().assignments(), &assignments);
    assert_eq!(stage.monitor().last_trace_snapshots(), latest.snapshots);
    let snapshot = stage.monitor().last_debug_snapshot().unwrap();
    assert_eq!(snapshot.frame_index, Some(88));
    assert!(snapshot.events.is_empty());
}

#[test]
fn policy_rejection_skips_callback_and_preserves_monitor_state() {
    let mut stage = stage(4, vec![room()]);
    let frame = fixture::frame_at(1.0, 21, vec![7]);
    let before_assignments = stage.monitor().assignments().clone();
    let before_traces = stage.monitor().last_trace_snapshots().to_vec();
    let mut callbacks = 0;
    let error = stage
        .observe(
            &frame,
            &[inside(7)],
            BedExitClocks {
                wall_time: None,
                observed_at: f64::NAN,
                snapshot_at: 1.0,
            },
            &mut |_| callbacks += 1,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        BedStageError::Policy(BedExitError::Rejected(_))
    ));
    assert_eq!(callbacks, 0);
    assert_eq!(stage.monitor().assignments(), &before_assignments);
    assert_eq!(stage.monitor().last_trace_snapshots(), before_traces);
    assert_eq!(stage.monitor().scoring(), BedExitScoring::default());
}

#[test]
fn fatal_partial_state_reports_the_exact_accepted_prefix_once() {
    // Arm outside the window so recovery proposals do not consume authority rows.
    let mut stage = fixture::windowed_stage(1, vec![room()]);
    arm_one(&mut stage, &[7, 8]);
    assert!(
        submit(&mut stage, 4.0, 5, vec![7, 8], vec![outside(7), outside(8)])
            .unwrap()
            .events
            .is_empty()
    );
    let frame = fixture::frame_at(5.0, 6, vec![7, 8]);
    let poses = [outside(7), outside(8)];
    let mut receipt = None;
    let error = stage
        .observe(
            &frame,
            &poses,
            BedExitClocks {
                wall_time: Some(fixture::wall_at(12)),
                observed_at: 5.0,
                snapshot_at: 5.0,
            },
            &mut |update| {
                assert!(receipt.is_none(), "partial prefix is reported once");
                receipt = Some(fixture::Receipt::from(update));
            },
        )
        .unwrap_err();
    let BedStageError::Policy(BedExitError::FatalPartialState { emitted_events, .. }) = error
    else {
        panic!("the second onset must hit the explicit episode limit");
    };
    let receipt = receipt.expect("non-empty accepted prefix must be observed before error");
    assert_eq!(emitted_events.len(), 1);
    assert_eq!(emitted_events[0].person_id, Some(7));
    assert_eq!(receipt.events, emitted_events);
    assert_eq!(receipt.frame, frame.identity);
    assert_eq!(receipt.frame_index, frame.frame_index);
    assert_eq!(receipt.time_sec, frame.time_sec);
    assert_eq!(receipt.snapshots, stage.monitor().last_trace_snapshots());
    assert_eq!(receipt.scoring, None);
    assert!(!receipt.completed);

    let poisoned_frame = fixture::frame_at(6.0, 7, vec![7, 8]);
    let mut callbacks = 0;
    let poisoned = stage
        .observe(
            &poisoned_frame,
            &poses,
            BedExitClocks {
                wall_time: None,
                observed_at: 6.0,
                snapshot_at: 6.0,
            },
            &mut |_| callbacks += 1,
        )
        .unwrap_err();
    assert!(matches!(
        poisoned,
        BedStageError::Policy(BedExitError::Poisoned(_))
    ));
    assert_eq!(callbacks, 0);
}
