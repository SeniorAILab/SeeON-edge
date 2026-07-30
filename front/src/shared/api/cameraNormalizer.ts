import {
  hasNullableString,
  isNonEmptyString,
  isNullableInteger,
  isRecord,
  pickNullableString,
  pickNumber,
  pickString,
} from '@/shared/api/normalizerFields';
import type { Camera, CameraRegistry, CameraStatus, CameraTestResult, DecodeBackend } from '@/shared/api/types';

function normalizeStatus(record: Record<string, unknown>): CameraStatus {
  const explicit = pickString(record, ['status']);
  if (explicit === 'online' || explicit === 'offline' || explicit === 'starting' || explicit === 'unknown') {
    return explicit;
  }
  if (record.online === true) return 'online';
  if (record.online === false) return 'offline';
  return 'unknown';
}

function normalizeDecodeBackend(record: Record<string, unknown>): DecodeBackend | string | null {
  for (const key of ['decode_backend', 'decodeBackend']) {
    const value = record[key];
    if (value === null || typeof value === 'string') {
      return value;
    }
  }
  return null;
}

function maskRtsp(value: string | null): string {
  if (!value) {
    return 'RTSP URL 비공개';
  }
  try {
    const url = new URL(value);
    if (!url.hostname) {
      return 'RTSP URL 비공개';
    }
    url.hostname = 'redacted-camera';
    url.username = url.username ? '***' : '';
    url.password = url.password ? '***' : '';
    return url.toString();
  } catch {
    return 'RTSP URL 비공개';
  }
}

export function normalizeCamera(value: unknown): Camera | null {
  if (!isRecord(value)) {
    return null;
  }

  const id = pickString(value, ['id', 'camera_id']);
  if (!id) {
    return null;
  }

  const rtspMasked = pickString(value, ['rtsp_url_masked', 'rtspUrlMasked']);
  const rtspPlain = pickNullableString(value, ['rtsp_url', 'rtspUrl']);
  const domains = isRecord(value.domains) ? Object.fromEntries(Object.entries(value.domains).map(([key, entry]) => [key, Boolean(entry)])) : null;

  return {
    id,
    label: pickString(value, ['label', 'name'], '이름 없는 카메라'),
    rtsp_url_masked: rtspMasked || maskRtsp(rtspPlain),
    space_id: pickNullableString(value, ['space_id', 'spaceId']),
    space_name: pickNullableString(value, ['space_name', 'spaceName']),
    floor_name: pickNullableString(value, ['floor_name', 'floorName']),
    backend_camera_id: pickNullableString(value, ['backend_camera_id', 'backendCameraId', 'facilityId']),
    status: normalizeStatus(value),
    created_at: pickNullableString(value, ['created_at', 'createdAt']),
    threshold: pickNumber(value, ['threshold']),
    domains,
    bed_count: pickNumber(value, ['bed_count', 'bedCount']),
    night_start: pickNullableString(value, ['night_start', 'nightStart']),
    night_end: pickNullableString(value, ['night_end', 'nightEnd']),
    decode_backend: normalizeDecodeBackend(value),
  };
}

export function normalizeCameraRegistry(value: unknown): CameraRegistry {
  if (
    !isRecord(value)
    || !Number.isInteger(value.registry_version)
    || (value.registry_version as number) < 0
    || !Array.isArray(value.cameras)
    || !value.cameras.every(isCameraResponse)
  ) {
    throw new Error('Invalid camera registry response');
  }
  return {
    registry_version: value.registry_version as number,
    cameras: value.cameras.map(normalizeCameraResponse),
  };
}

export function normalizeCameraResponse(value: unknown): Camera {
  if (!isCameraResponse(value)) {
    throw new Error('Invalid camera response');
  }
  return normalizeCamera(value) as Camera;
}

function isCameraResponse(value: unknown): value is Record<string, unknown> {
  if (!isRecord(value)) return false;
  const status = value.status;
  return isNonEmptyString(value.id)
    && isNonEmptyString(value.label)
    && isNonEmptyString(value.rtsp_url_masked)
    && (status === 'online' || status === 'offline' || status === 'starting' || status === 'unknown')
    && (!('mapping_pending' in value) || typeof value.mapping_pending === 'boolean')
    && hasNullableString(value, 'space_id')
    && hasNullableString(value, 'backend_camera_id')
    && hasNullableString(value, 'decode_backend')
    && hasNullableString(value, 'created_at')
    && hasNullableString(value, 'space_name')
    && hasNullableString(value, 'floor_name');
}

export function normalizeCameraTestResult(value: unknown): CameraTestResult {
  if (
    !isRecord(value)
    || typeof value.ok !== 'boolean'
    || ('error_class' in value && value.error_class !== null && value.error_class !== 'timeout' && value.error_class !== 'decode' && value.error_class !== 'auth')
    || ('width' in value && !isNullableInteger(value.width))
    || ('height' in value && !isNullableInteger(value.height))
  ) {
    throw new Error('Invalid camera test response');
  }
  const result: CameraTestResult = { ok: value.ok };
  if (value.error_class === 'timeout' || value.error_class === 'decode' || value.error_class === 'auth') result.error_class = value.error_class;
  if (typeof value.width === 'number') result.width = value.width;
  if (typeof value.height === 'number') result.height = value.height;
  return result;
}
