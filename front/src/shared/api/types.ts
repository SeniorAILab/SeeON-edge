export type CameraStatus = 'online' | 'offline' | 'starting' | 'unknown';

export type DecodeBackend = 'auto' | 'nvdec' | 'opencv' | 'cpu';

export type Camera = {
  id: string;
  label: string;
  rtsp_url_masked: string;
  space_id: string | null;
  space_name: string | null;
  floor_name: string | null;
  backend_camera_id: string | null;
  status: CameraStatus;
  created_at: string | null;
  threshold?: number | null;
  domains?: Record<string, boolean> | null;
  bed_count?: number | null;
  night_start?: string | null;
  night_end?: string | null;
  decode_backend?: DecodeBackend | string | null;
};

export type CameraRegistry = {
  registry_version: number;
  cameras: Camera[];
};

export type HeartbeatStatus = 'online' | 'stale' | 'never_seen';

export type CameraHeartbeat = {
  camera_id: string;
  facility_id: string | null;
  status: HeartbeatStatus;
  last_heartbeat_at: number | null;
  age_sec: number | null;
  config_version: number | null;
};

export type RuntimeDecodeDiagnostics = {
  requested: string | null;
  selected: string | null;
  fallback_count: number | null;
  last_reason: string | null;
  updated_at_sec: number | null;
};

export type RuntimeCamera = {
  camera_id: string;
  decode: RuntimeDecodeDiagnostics;
  measured_fps?: number | null;
};

export type RuntimeGpuDiagnostics = {
  nvml_available: boolean | null;
  cuda_context_ok: boolean | null;
  driver_version: string | null;
  device_name: string | null;
  captured_at_sec: number | null;
};

export type RuntimeWorkerDiagnostics = {
  alive: boolean | null;
  pid: number | null;
  started_at_sec: number | null;
};

export type RuntimeLatencyDiagnostics = {
  first_attempt_samples: number | null;
  max_sec: number | null;
  since_sec: number | null;
};

export type RuntimeClipRecorder = {
  available: boolean | null;
  dropped_frames: number | null;
  dropped_events: number | null;
  failed_writes: number | null;
  finalized_clips: number | null;
  video_unavailable_clips: number | null;
  active_clips: number | null;
  encoder: string | null;
};

export type RuntimeFacility = {
  facility_id: string;
  generation: number | null;
  seq: number | null;
  received_at: number | null;
  stale: boolean | null;
  cameras: RuntimeCamera[];
  clip_recorder: RuntimeClipRecorder | null;
  gpu?: RuntimeGpuDiagnostics | null;
  worker?: RuntimeWorkerDiagnostics | null;
  latency?: RuntimeLatencyDiagnostics | null;
};

export type StatusSnapshot = {
  cameras: Record<string, CameraHeartbeat>;
  stale_after_sec: number | null;
  runtime: {
    facilities: Record<string, RuntimeFacility>;
    stale_after_sec: number | null;
  };
};

export type CameraInput = {
  label: string;
  rtsp_url: string;
  space_id?: string;
};

export type CameraPatchInput = Partial<CameraInput> & {
  detectionSettings?: {
    threshold: number;
    domains: Record<string, boolean>;
    bedCount: number;
    nightWindow: { start: string; end: string };
  };
  decode_backend?: DecodeBackend;
};

export type CameraTestResult = {
  ok: boolean;
  error_class?: 'timeout' | 'decode' | 'auth' | string;
  width?: number;
  height?: number;
};

export type SystemSnapshot = {
  updated_at?: string | null;
  backend: {
    configured: boolean;
    reachable: boolean | null;
    last_ok_at: string | null;
  };
  version: string | null;
  image_digests?: {
    ml_api: string | null;
    ml_worker: string | null;
  };
  storage?: {
    clip_store?: {
      total_bytes: number | null;
      used_bytes: number | null;
      used_pct: number | null;
    };
  };
};

export type ClipLabel = 'TRUE_POSITIVE' | 'FALSE_POSITIVE' | 'UNREVIEWED';

export type Clip = {
  id: string;
  camera_id: string | null;
  camera_label: string;
  event_type: string;
  created_at: string | null;
  label: ClipLabel | null;
  reviewer?: string | null;
  reviewed_at?: string | null;
  reviewState?: 'unknown' | 'confirmed';
  video_path: string;
  video_available: boolean;
  video_error: string | null;
};
