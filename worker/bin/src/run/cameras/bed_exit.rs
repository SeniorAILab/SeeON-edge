//! Root-owned bed-exit camera policy, identity, and raster resource admission.

use seeon_worker::bed_exit::{
    BedExitCapacities, BedExitConfig, BedExitMonitor, DEFAULT_STALE_AFTER_SEC,
};
use seeon_worker::detection_window::DetectionWindow;

use crate::policy::bed_exit::{BedStage, BedStageError};
use crate::policy::bed_exit_input::BedInputProducer;
use crate::relay::cameras::RuntimeCamera;
use crate::relay::cameras::policies::{EffectivePolicy, PolicyBundle, PolicyValues};

const BED_EXIT_MODULE: &str = "bed_exit";
const BED_EXIT_MODULE_VERSION: i128 = 1;
const INITIAL_NATIVE_EPOCH: u64 = 1;
const INITIAL_NATIVE_GENERATION: u64 = 1;

/// One immutable process-start budget shared by the admitted camera roster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BedPolicyBudget {
    pub capacities: BedExitCapacities,
    pub mask_cells: usize,
}

/// Prefix cells are u64-sized entries; the entire roster may retain at most 512 MiB.
pub const BED_EXIT_MASK_CELLS: usize = 64 * 1024 * 1024;

/// Fixed implementation bounds, owned by the runtime root and not operator policy.
pub const DEFAULT_BED_POLICY_BUDGET: BedPolicyBudget = BedPolicyBudget {
    capacities: BedExitCapacities {
        tracks: 150,
        observations: 150,
        beds: 16,
        pose_features: 150,
        polygon_points: 16,
        episodes: 4096,
    },
    mask_cells: BED_EXIT_MASK_CELLS,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ResolvedBedExitPolicy {
    min_containment: f64,
    hold_frames: usize,
    grace_frames: u64,
}

pub(super) fn resolve_policy(
    policies: &PolicyBundle,
    camera_id: &str,
) -> Result<ResolvedBedExitPolicy, super::CameraPolicyError> {
    let selected = admitted_policy(policies, camera_id)?;
    let PolicyValues::BedExitV1 {
        min_containment,
        hold_frames,
        grace_frames,
    } = selected.values
    else {
        return Err(super::CameraPolicyError::BedPolicyShape);
    };
    Ok(ResolvedBedExitPolicy {
        min_containment,
        hold_frames: usize::try_from(hold_frames)
            .map_err(|_| super::CameraPolicyError::BedPolicyBound)?,
        grace_frames: u64::try_from(grace_frames)
            .map_err(|_| super::CameraPolicyError::BedPolicyBound)?,
    })
}

pub(super) fn admitted_policy<'a>(
    policies: &'a PolicyBundle,
    camera_id: &str,
) -> Result<&'a EffectivePolicy, super::CameraPolicyError> {
    let selected = policies
        .cameras
        .get(camera_id)
        .and_then(|camera| camera.get(BED_EXIT_MODULE))
        .or_else(|| policies.defaults.get(BED_EXIT_MODULE))
        .ok_or(super::CameraPolicyError::BedPolicyMissing)?;
    if selected.module_id != BED_EXIT_MODULE
        || selected.module_version != BED_EXIT_MODULE_VERSION
        || selected.schema_id != "bed_exit.policy"
        || selected.schema_version != 1
        || !matches!(selected.values, PolicyValues::BedExitV1 { .. })
    {
        return Err(super::CameraPolicyError::BedPolicyShape);
    }
    Ok(selected)
}

pub(super) fn build_stage(
    camera: &RuntimeCamera,
    boot_id: &str,
    resolved: ResolvedBedExitPolicy,
    night_window: Option<DetectionWindow>,
    capacities: BedExitCapacities,
    remaining_mask_cells: usize,
) -> Result<(BedStage, usize), super::CameraPolicyError> {
    let inputs = BedInputProducer::new(camera, capacities, remaining_mask_cells)
        .map_err(|error| super::CameraPolicyError::BedStage(BedStageError::Input(error)))?;
    let allocated_cells = inputs.allocated_cells();
    for (bed_index, reason) in inputs.fallbacks() {
        eprintln!(
            "ml-worker: bed containment fallback camera_id={} bed_index={} reason={reason:?} method=aabb",
            camera.camera_id, bed_index
        );
    }
    let monitor = BedExitMonitor::new(
        BedExitConfig {
            camera_id: camera.camera_id.clone(),
            facility_id: camera.facility_id.clone(),
            min_containment: resolved.min_containment,
            hold_frames: resolved.hold_frames,
            grace_frames: resolved.grace_frames,
            night_window,
            in_bed_dwell_sec: 3.0,
            outside_dwell_sec: 2.0,
        },
        boot_id,
        INITIAL_NATIVE_EPOCH.to_string(),
        INITIAL_NATIVE_GENERATION,
        capacities,
        DEFAULT_STALE_AFTER_SEC,
    )
    .map_err(super::CameraPolicyError::BedMonitor)?;
    Ok((BedStage::new(monitor, inputs), allocated_cells))
}
