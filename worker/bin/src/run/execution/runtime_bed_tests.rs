use super::*;
use crate::policy::bed_exit::{BedStage, BedStageError};
use crate::policy::ingest::{Frame, ObservedPose};
use seeon_worker::bed_exit::{BedExitClocks, BedExitError};

#[path = "runtime_bed_fixture.rs"]
mod fixture;
use fixture::stage;

fn update(
    stage: &mut BedStage,
    sink: &mut LiveSink,
    sequence: u64,
    tracks: &[u64],
) -> Result<(), BedStageError> {
    sink.preflight_bed_decision()
        .expect("bounded setup receipt");
    let left = if sequence < 6 { 2.0 } else { 24.0 };
    let frame = Frame {
        identity: seeon_deepstream_native::FrameIdentity {
            sequence,
            frame_number: sequence as i64,
            pts_ns: sequence * 1_000_000_000,
            pts_valid: 1,
            source_id: 3,
            source_width: 32,
            source_height: 32,
            analysis_width: 32,
            analysis_height: 32,
            binding: seeon_deepstream_native::MediaBinding {
                token: 4,
                generation: 1,
                epoch: 1,
            },
            ..Default::default()
        },
        width: 32,
        height: 32,
        live_track_ids: tracks.to_vec(),
        rows: BTreeMap::new(),
        time_sec: Some(sequence as f64),
        frame_index: sequence as i64,
    };
    let poses: Vec<_> = tracks
        .iter()
        .map(|&track_id| ObservedPose {
            track_id,
            bbox: [left, 2.0, left + 4.0, 8.0],
            confidence: 0.9,
            keypoints: vec![[left + 2.0, 5.0, 1.0]; 17],
        })
        .collect();
    stage.observe(
        &frame,
        &poses,
        BedExitClocks {
            wall_time: Some(
                seeon_worker::detection_window::AwareDateTime::from_utc_system_time(
                    std::time::UNIX_EPOCH
                        + Duration::from_secs(if sequence < 7 { 0 } else { 43_200 }),
                )
                .unwrap(),
            ),
            observed_at: sequence as f64,
            snapshot_at: sequence as f64,
        },
        &mut |receipt| sink.bed_decision(receipt).expect("bounded test receipt"),
    )
}

#[test]
fn bed_receipt_does_not_consume_held_fall_scores_or_invent_generation() {
    let mut stage = stage(8, false);
    let mut sink = LiveSink::new(Arc::new(crate::seam::SystemClock::new()));
    let frame = seeon_deepstream_native::FrameIdentity {
        source_id: 3,
        sequence: 0,
        pts_valid: 1,
        source_width: 32,
        source_height: 32,
        analysis_width: 32,
        analysis_height: 32,
        binding: seeon_deepstream_native::MediaBinding {
            token: 4,
            generation: 1,
            epoch: 1,
        },
        ..Default::default()
    };
    sink.scores.push(super::super::output::HeldScore {
        frame,
        track_id: 7,
        logit: 0.75,
        evidence: None,
    });
    update(&mut stage, &mut sink, 0, &[7]).unwrap();
    assert_eq!(sink.scores.len(), 1);
    assert_eq!(sink.scores[0].frame, frame);
    assert_eq!(sink.scores[0].logit, 0.75);
    assert_eq!(sink.ready.len(), 1);
    assert!(sink.ready[0].scores.is_empty());
    assert!(!sink.ready[0].decisions.is_empty());
    assert!(
        sink.ready[0]
            .decisions
            .iter()
            .all(|decision| decision.generation.is_none())
    );
}

#[test]
fn fatal_bed_prefix_is_retained_without_claiming_all_traces_completed() {
    // Arming outside the window leaves authority capacity for the two onsets.
    let mut stage = stage(1, true);
    let mut sink = LiveSink::new(Arc::new(crate::seam::SystemClock::new()));
    for sequence in 0..7 {
        update(&mut stage, &mut sink, sequence, &[7, 8]).unwrap();
        assert!(sink.triggered.is_empty());
        sink.ready.clear();
    }
    let error = update(&mut stage, &mut sink, 7, &[7, 8]).unwrap_err();
    let BedStageError::Policy(BedExitError::FatalPartialState { emitted_events, .. }) = error
    else {
        panic!("second onset must exhaust the real episode capacity");
    };
    assert_eq!(emitted_events.len(), 1);
    assert_eq!(sink.triggered.len(), 1);
    assert_eq!(sink.triggered[0].event, emitted_events[0]);
    assert_eq!(sink.triggered[0].frame.sequence, 7);
    assert!(sink.ready.iter().all(|batch| batch.decisions.is_empty()));
    assert!(sink.scores.is_empty());
}

#[test]
fn sink_capacity_refusal_precedes_accepting_a_new_bed_onset() {
    let mut stage = stage(8, false);
    let clock = Arc::new(crate::seam::SystemClock::new());
    let mut sink = LiveSink::new(clock.clone());
    for sequence in 0..7 {
        update(&mut stage, &mut sink, sequence, &[7]).unwrap();
        assert!(sink.triggered.is_empty());
        sink.ready.clear();
    }
    let mut row = [0.0; 57];
    row[..5].copy_from_slice(&[24.0, 2.0, 28.0, 8.0, 0.9]);
    for coordinate in &mut row[..4] {
        *coordinate *= 20.0;
    }
    for joint in 0..17 {
        row[6 + 3 * joint..9 + 3 * joint].copy_from_slice(&[520.0, 100.0, 1.0]);
    }
    let packet = crate::msg::PosePacket {
        frame: seeon_deepstream_native::FrameIdentity {
            source_id: 3,
            sequence: 7,
            frame_number: 7,
            pts_ns: 7_000_000_000,
            pts_valid: 1,
            source_width: 32,
            source_height: 32,
            analysis_width: 32,
            analysis_height: 32,
            binding: seeon_deepstream_native::MediaBinding {
                token: 4,
                generation: 1,
                epoch: 1,
            },
            ..Default::default()
        },
        tensor_present: true,
        rows: vec![row],
        objects: vec![seeon_deepstream_native::TrackedObject {
            track_id: 7,
            left: 24.0,
            top: 2.0,
            width: 4.0,
            height: 6.0,
            confidence: 0.9,
        }],
    };
    let mut pump = crate::run::pump::PolicyPump::new(
        vec![crate::run::pump::CameraPolicy {
            source_id: 3,
            stage: None,
            bed_stage: Some(stage),
        }],
        None,
        clock,
    )
    .unwrap();
    for _ in 0..DECISION_BATCH_RETENTION {
        sink.ready.push(FrameScores {
            frame: packet.frame,
            scores: Vec::new(),
            decisions: Vec::new(),
        });
    }
    assert!(pump.observe(&packet, None, &mut sink).is_err());
    assert!(sink.triggered.is_empty());
    sink.ready.clear();
    pump.observe(&packet, None, &mut sink).unwrap();
    assert_eq!(sink.triggered.len(), 1);
    assert_eq!(sink.triggered[0].event.person_id, Some(7));
    assert_eq!(sink.triggered[0].event.time_sec, 7.0);
}
