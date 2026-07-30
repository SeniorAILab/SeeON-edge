import {
  isRecord,
  pickBoolean,
  pickNonNegativeNumber,
  pickNullableString,
  pickString,
} from '@/shared/api/normalizerFields';
import type {
  CameraHeartbeat,
  RuntimeCamera,
  RuntimeClipRecorder,
  RuntimeFacility,
  RuntimeGpuDiagnostics,
  RuntimeLatencyDiagnostics,
  RuntimeWorkerDiagnostics,
  StatusSnapshot,
} from '@/shared/api/types';

function normalizeHeartbeat(value: unknown): CameraHeartbeat | null {
  if (!isRecord(value)) return null;
  const cameraId = pickString(value, ['camera_id', 'cameraId']);
  const status = pickString(value, ['status']);
  if (!cameraId || (status !== 'online' && status !== 'stale' && status !== 'never_seen')) return null;
  return {
    camera_id: cameraId,
    facility_id: pickNullableString(value, ['facility_id', 'facilityId']),
    status,
    last_heartbeat_at: pickNonNegativeNumber(value, ['last_heartbeat_at', 'lastHeartbeatAt']),
    age_sec: pickNonNegativeNumber(value, ['age_sec', 'ageSec']),
    config_version: pickNonNegativeNumber(value, ['config_version', 'configVersion']),
  };
}

function normalizeRuntimeCamera(value: unknown): RuntimeCamera | null {
  if (!isRecord(value)) return null;
  const cameraId = pickString(value, ['camera_id', 'cameraId']);
  if (!cameraId) return null;
  const decode = isRecord(value.decode) ? value.decode : {};
  return {
    camera_id: cameraId,
    decode: {
      requested: pickNullableString(decode, ['requested']),
      selected: pickNullableString(decode, ['selected']),
      fallback_count: pickNonNegativeNumber(decode, ['fallback_count', 'fallbackCount']),
      last_reason: pickNullableString(decode, ['last_reason', 'lastReason']),
      updated_at_sec: pickNonNegativeNumber(decode, ['updated_at_sec', 'updatedAtSec']),
    },
    ...(('measured_fps' in value || 'measuredFps' in value)
      ? { measured_fps: pickNonNegativeNumber(value, ['measured_fps', 'measuredFps']) }
      : {}),
  };
}

function normalizeRuntimeGpu(value: unknown): RuntimeGpuDiagnostics | null {
  if (!isRecord(value)) return null;
  return {
    nvml_available: pickBoolean(value, ['nvml_available', 'nvmlAvailable']),
    cuda_context_ok: pickBoolean(value, ['cuda_context_ok', 'cudaContextOk']),
    driver_version: pickNullableString(value, ['driver_version', 'driverVersion']),
    device_name: pickNullableString(value, ['device_name', 'deviceName']),
    captured_at_sec: pickNonNegativeNumber(value, ['captured_at_sec', 'capturedAtSec']),
  };
}

function normalizeRuntimeWorker(value: unknown): RuntimeWorkerDiagnostics | null {
  if (!isRecord(value)) return null;
  const pid = pickNonNegativeNumber(value, ['pid']);
  return {
    alive: pickBoolean(value, ['alive']),
    pid: pid !== null && Number.isInteger(pid) ? pid : null,
    started_at_sec: pickNonNegativeNumber(value, ['started_at_sec', 'startedAtSec']),
  };
}

function normalizeRuntimeLatency(value: unknown): RuntimeLatencyDiagnostics | null {
  if (!isRecord(value)) return null;
  return {
    first_attempt_samples: pickNonNegativeNumber(value, ['first_attempt_samples', 'firstAttemptSamples']),
    max_sec: pickNonNegativeNumber(value, ['max_sec', 'maxSec']),
    since_sec: pickNonNegativeNumber(value, ['since_sec', 'sinceSec']),
  };
}

function normalizeRuntimeClipRecorder(value: unknown): RuntimeClipRecorder | null {
  if (!isRecord(value)) return null;
  return {
    available: pickBoolean(value, ['available']),
    dropped_frames: pickNonNegativeNumber(value, ['dropped_frames', 'droppedFrames']),
    dropped_events: pickNonNegativeNumber(value, ['dropped_events', 'droppedEvents']),
    failed_writes: pickNonNegativeNumber(value, ['failed_writes', 'failedWrites']),
    finalized_clips: pickNonNegativeNumber(value, ['finalized_clips', 'finalizedClips']),
    video_unavailable_clips: pickNonNegativeNumber(value, ['video_unavailable_clips', 'videoUnavailableClips']),
    active_clips: pickNonNegativeNumber(value, ['active_clips', 'activeClips']),
    encoder: pickNullableString(value, ['encoder']),
  };
}

function normalizeRuntimeFacility(value: unknown): RuntimeFacility | null {
  if (!isRecord(value)) return null;
  const facilityId = pickString(value, ['facility_id', 'facilityId']);
  if (!facilityId) return null;
  return {
    facility_id: facilityId,
    generation: pickNonNegativeNumber(value, ['generation']),
    seq: pickNonNegativeNumber(value, ['seq']),
    received_at: pickNonNegativeNumber(value, ['received_at', 'receivedAt']),
    stale: pickBoolean(value, ['stale']),
    cameras: Array.isArray(value.cameras) ? value.cameras.map(normalizeRuntimeCamera).filter((camera): camera is RuntimeCamera => camera !== null) : [],
    clip_recorder: normalizeRuntimeClipRecorder(value.clip_recorder),
    ...(('gpu' in value) ? { gpu: normalizeRuntimeGpu(value.gpu) } : {}),
    ...(('worker' in value) ? { worker: normalizeRuntimeWorker(value.worker) } : {}),
    ...(('latency' in value) ? { latency: normalizeRuntimeLatency(value.latency) } : {}),
  };
}

export function normalizeStatusSnapshot(value: unknown): StatusSnapshot {
  const record = isRecord(value) ? value : null;
  const rawCameras = record?.cameras;
  const rawRuntime = record?.runtime;
  const rawFacilities = isRecord(rawRuntime) ? rawRuntime.facilities : null;
  if (!record || !isRecord(rawCameras) || !isRecord(rawRuntime) || !isRecord(rawFacilities)) {
    throw new Error('Invalid status response');
  }
  const cameras: Record<string, CameraHeartbeat> = {};
  Object.entries(rawCameras).forEach(([key, entry]) => {
    const heartbeat = normalizeHeartbeat(entry);
    if (heartbeat) cameras[key] = heartbeat;
  });

  const facilities: Record<string, RuntimeFacility> = {};
  Object.entries(rawFacilities).forEach(([key, entry]) => {
    const facility = normalizeRuntimeFacility(entry);
    if (facility) facilities[key] = facility;
  });
  return {
    cameras,
    stale_after_sec: pickNonNegativeNumber(record, ['stale_after_sec', 'staleAfterSec']),
    runtime: {
      facilities,
      stale_after_sec: pickNonNegativeNumber(rawRuntime, ['stale_after_sec', 'staleAfterSec']),
    },
  };
}
