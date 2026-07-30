import { requestJson } from '@/shared/api/http';
import { normalizeCameraRegistry, normalizeCameraResponse, normalizeCameraTestResult, normalizeClipsResponse, normalizeStatusSnapshot, normalizeSystemSnapshot } from '@/shared/api/normalizers';
import { getApiBase, getCameraSnapshotUrl, getCameraStreamUrl } from '@/shared/api/session';
import type { Camera, CameraInput, CameraPatchInput, CameraRegistry, CameraTestResult, Clip, ClipLabel, DecodeBackend, StatusSnapshot, SystemSnapshot } from '@/shared/api/types';

export type {
  Camera,
  CameraInput,
  CameraPatchInput,
  CameraRegistry,
  CameraStatus,
  CameraHeartbeat,
  CameraTestResult,
  Clip,
  ClipLabel,
  DecodeBackend,
  HeartbeatStatus,
  RuntimeCamera,
  RuntimeClipRecorder,
  RuntimeDecodeDiagnostics,
  RuntimeFacility,
  StatusSnapshot,
  SystemSnapshot,
} from '@/shared/api/types';

export {
  getApiBase,
  getCameraSnapshotUrl,
  getCameraStreamUrl,
} from '@/shared/api/session';

export async function loginDashboard(username: string, password: string): Promise<void> {
  await requestJson('/auth/session', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  });
}

export async function fetchDashboardSession(): Promise<void> {
  await requestJson('/auth/session');
}

export async function logoutDashboard(): Promise<void> {
  await requestJson('/auth/session', { method: 'DELETE' });
}

type ClipLabelOverlay = Pick<Clip, 'label' | 'reviewer' | 'reviewed_at'>;
const clipLabelOverlay = new Map<string, ClipLabelOverlay>();

export function cameraMediaId(camera: Pick<Camera, 'id' | 'backend_camera_id'>): string {
  return camera.backend_camera_id ?? camera.id;
}

function cameraBody(input: CameraInput | CameraPatchInput): string {
  const body: Record<string, unknown> = {};
  if (input.label !== undefined) {
    body.label = input.label.trim();
  }
  if (input.rtsp_url !== undefined) {
    body.rtsp_url = input.rtsp_url.trim();
  }
  if (input.space_id !== undefined) {
    body.space_id = input.space_id.trim();
  }
  if ('detectionSettings' in input && input.detectionSettings) {
    body.detectionSettings = input.detectionSettings;
  }
  if ('decode_backend' in input && input.decode_backend !== undefined) {
    body.decode_backend = input.decode_backend;
  }
  return JSON.stringify(body);
}

export async function fetchCameras(signal?: AbortSignal): Promise<CameraRegistry> {
  return normalizeCameraRegistry(await requestJson('/cameras', { signal }));
}

export async function createCamera(input: CameraInput): Promise<Camera> {
  return normalizeCameraResponse(await requestJson('/cameras', { method: 'POST', body: cameraBody(input) }));
}

export async function updateCamera(cameraId: string, input: CameraPatchInput): Promise<Camera> {
  return normalizeCameraResponse(
    await requestJson(`/cameras/${encodeURIComponent(cameraId)}`, { method: 'PATCH', body: cameraBody(input) }),
  );
}

export async function updateCameraDecodeBackend(cameraId: string, backend: DecodeBackend): Promise<Camera> {
  return updateCamera(cameraId, { decode_backend: backend });
}

export async function deleteCamera(cameraId: string): Promise<void> {
  await requestJson(`/cameras/${encodeURIComponent(cameraId)}`, { method: 'DELETE' });
}

export async function testCamera(cameraId: string): Promise<CameraTestResult> {
  return normalizeCameraTestResult(await requestJson(`/cameras/${encodeURIComponent(cameraId)}/test`, { method: 'POST' }));
}

export async function fetchStatus(signal?: AbortSignal): Promise<StatusSnapshot> {
  return normalizeStatusSnapshot(await requestJson('/status', { signal }));
}

export async function fetchSystem(signal?: AbortSignal): Promise<SystemSnapshot> {
  return normalizeSystemSnapshot(await requestJson('/system', { signal }));
}

export async function fetchClips(cameraId?: string, signal?: AbortSignal): Promise<Clip[]> {
  const query = cameraId?.trim() ? `?camera_id=${encodeURIComponent(cameraId.trim())}` : '';
  const value = await requestJson(`/clips${query}`, { signal });
  return normalizeClipsResponse(value).map(applyClipLabelOverlay);
}

export async function labelClip(existing: Clip, label: ClipLabel): Promise<Clip> {
  const value = await requestJson(`/clips/${encodeURIComponent(existing.id)}/label`, {
      method: 'PUT',
      body: JSON.stringify({ label: label === 'UNREVIEWED' ? null : label }),
    });
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error('Invalid clip response');
  }
  const response = value as Record<string, unknown>;
  if ((response.clip_id !== existing.id && response.id !== existing.id)
    || (response.label !== null && response.label !== 'TRUE_POSITIVE' && response.label !== 'FALSE_POSITIVE')
    || typeof response.reviewer !== 'string' || !response.reviewer
    || typeof response.reviewed_at !== 'string' || !response.reviewed_at) {
    throw new Error('Invalid clip response');
  }
  const overlay: ClipLabelOverlay = {
    label: response.label,
    reviewer: response.reviewer,
    reviewed_at: response.reviewed_at,
  };
  clipLabelOverlay.set(existing.id, overlay);
  return { ...existing, ...overlay, reviewState: 'confirmed' };
}

function applyClipLabelOverlay(clip: Clip): Clip {
  const overlay = clipLabelOverlay.get(clip.id);
  return overlay ? { ...clip, ...overlay, reviewState: 'confirmed' } : clip;
}

export function clearClipLabelOverlay(): void {
  clipLabelOverlay.clear();
}
