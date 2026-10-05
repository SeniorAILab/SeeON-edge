use std::collections::BTreeMap;

use seeon_worker::bed_containment::{BedContainment, RasterLimits};
use seeon_worker::bed_exit::{
    AssignmentSnapshot, BedExitInput, BedRegionCacheState, BedRegionDebugSnapshot, BoundingBox,
    PriorBoxContainments,
};

use super::regions;
use super::{BedInputError, BedInputProducer, capacity, reserve};
use crate::policy::ingest::{Frame, ObservedPose};

pub(super) fn build(
    producer: &mut BedInputProducer,
    frame: &Frame,
    observed: &[ObservedPose],
    assignments: &BTreeMap<u64, AssignmentSnapshot>,
) -> Result<BedExitInput, BedInputError> {
    let limits = producer.capacities;
    capacity("observations", observed.len(), limits.observations)?;
    capacity("pose_features", observed.len(), limits.pose_features)?;
    capacity("live_track_ids", frame.live_track_ids.len(), limits.tracks)?;
    capacity("assignments", assignments.len(), limits.tracks)?;
    observed
        .len()
        .checked_mul(producer.regions.beds.len())
        .ok_or(BedInputError::Overflow("containments"))?;
    observed
        .len()
        .checked_mul(assignments.len())
        .ok_or(BedInputError::Overflow("prior_box_containments"))?;
    let frame_size = [frame.width, frame.height];
    let rebuild = producer
        .pose_geometry
        .as_ref()
        .is_none_or(|(cached_size, _)| *cached_size != frame_size);
    if rebuild {
        producer.pose_geometry = None;
        let geometry = regions::pose_geometry(
            &producer.regions.beds,
            frame_size,
            producer.polygon_size,
            limits,
        )?;
        producer.pose_geometry = Some((frame_size, geometry));
    }
    let geometry = producer
        .pose_geometry
        .as_ref()
        .map(|(_, geometry)| geometry)
        .ok_or(BedInputError::Conversion("pose_geometry_cache"))?;

    let mut person_boxes = Vec::new();
    let mut track_ids = Vec::new();
    let mut bed_pose_features = Vec::new();
    let mut containments = Vec::new();
    reserve(&mut person_boxes, observed.len(), "person_boxes")?;
    reserve(&mut track_ids, observed.len(), "track_ids")?;
    reserve(&mut bed_pose_features, observed.len(), "pose_features")?;
    reserve(&mut containments, observed.len(), "containment_rows")?;
    for pose in observed {
        let person = person_box(pose)?;
        track_ids.push(Some(pose.track_id));
        bed_pose_features.push(geometry.features(pose.track_id, &pose.keypoints)?);
        let mut row = Vec::new();
        reserve(
            &mut row,
            producer.regions.masks.len(),
            "containment_columns",
        )?;
        for mask in &producer.regions.masks {
            row.push(mask.ratio(&person)?);
        }
        person_boxes.push(person);
        containments.push(row);
    }
    let prior_box_containments = prior_rows(producer, assignments, &person_boxes)?;
    let mut live_track_ids = Vec::new();
    reserve(
        &mut live_track_ids,
        frame.live_track_ids.len(),
        "live_track_ids",
    )?;
    live_track_ids.extend_from_slice(&frame.live_track_ids);
    let bed_region = BedRegionDebugSnapshot {
        source: if producer.regions.beds.is_empty() {
            BedRegionCacheState::Empty
        } else {
            BedRegionCacheState::Fresh
        },
        empty_cycles: 0,
    };
    Ok(BedExitInput {
        person_boxes,
        bed_boxes: copy_beds(&producer.regions.beds, producer.capacities.polygon_points)?,
        track_ids,
        live_track_ids,
        bed_pose_features,
        containments,
        prior_box_containments,
        time_sec: frame.time_sec,
        frame_index: frame.frame_index,
        bed_region,
    })
}

fn copy_beds(beds: &[BoundingBox], max_vertices: usize) -> Result<Vec<BoundingBox>, BedInputError> {
    let mut result = Vec::new();
    reserve(&mut result, beds.len(), "bed_boxes")?;
    for bed in beds {
        result.push(copy_box(bed, max_vertices)?);
    }
    Ok(result)
}

fn person_box(pose: &ObservedPose) -> Result<BoundingBox, BedInputError> {
    if !pose.confidence.is_finite() {
        return Err(BedInputError::Conversion("person_confidence"));
    }
    Ok(BoundingBox {
        x1: integer(pose.bbox[0], "person_bbox")?,
        y1: integer(pose.bbox[1], "person_bbox")?,
        x2: integer(pose.bbox[2], "person_bbox")?,
        y2: integer(pose.bbox[3], "person_bbox")?,
        confidence: pose.confidence,
        polygon: None,
    })
}

fn integer(value: f64, field: &'static str) -> Result<i64, BedInputError> {
    const MIN: f64 = -9_223_372_036_854_775_808.0;
    const MAX_EXCLUSIVE: f64 = 9_223_372_036_854_775_808.0;
    if !value.is_finite() || value.fract() != 0.0 || value < MIN || value >= MAX_EXCLUSIVE {
        return Err(BedInputError::Conversion(field));
    }
    Ok(value as i64)
}

fn prior_rows(
    producer: &BedInputProducer,
    assignments: &BTreeMap<u64, AssignmentSnapshot>,
    people: &[BoundingBox],
) -> Result<Vec<PriorBoxContainments>, BedInputError> {
    let count = assignments
        .values()
        .filter(|assignment| assignment.last_box.is_some())
        .count();
    capacity("prior_box_rows", count, producer.capacities.tracks)?;
    let mut rows = Vec::new();
    reserve(&mut rows, count, "prior_box_rows")?;
    let mask_cells = producer
        .regions
        .mask_cell_limit
        .checked_sub(producer.regions.allocated_cells)
        .ok_or(BedInputError::Overflow("mask_cells"))?;
    for (&track_id, assignment) in assignments {
        let Some(previous) = assignment.last_box.as_ref() else {
            continue;
        };
        let previous_box = copy_box(previous, producer.capacities.polygon_points)?;
        let mask = BedContainment::new(
            &previous_box,
            RasterLimits {
                max_cells: mask_cells,
                max_vertices: producer.capacities.polygon_points,
            },
        )?;
        let mut ratios = Vec::new();
        reserve(&mut ratios, people.len(), "overlap_columns")?;
        for person in people {
            ratios.push(mask.ratio(person)?);
        }
        rows.push(PriorBoxContainments {
            previous_track_id: track_id,
            previous_box,
            ratios,
        });
    }
    Ok(rows)
}

fn copy_box(value: &BoundingBox, max_vertices: usize) -> Result<BoundingBox, BedInputError> {
    let polygon = match &value.polygon {
        None => None,
        Some(points) => {
            capacity("polygon_points", points.len(), max_vertices)?;
            let mut copy = Vec::new();
            reserve(&mut copy, points.len(), "box_polygon")?;
            copy.extend_from_slice(points);
            Some(copy)
        }
    };
    Ok(BoundingBox {
        x1: value.x1,
        y1: value.y1,
        x2: value.x2,
        y2: value.y2,
        confidence: value.confidence,
        polygon,
    })
}
