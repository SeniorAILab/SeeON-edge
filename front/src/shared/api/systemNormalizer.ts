import {
  isNullableBoolean,
  isNullableFiniteNumber,
  isNullableInteger,
  isNullableString,
  isRecord,
  pickNullableString,
} from '@/shared/api/normalizerFields';
import type { SystemSnapshot } from '@/shared/api/types';

function normalizeVersion(record: Record<string, unknown>): string | null {
  const version = pickNullableString(record, ['version']);
  return version?.trim().toLowerCase() === 'unknown' ? null : version;
}

export function normalizeSystemSnapshot(value: unknown): SystemSnapshot {
  const record = isRecord(value) ? value : null;
  const backend = record && isRecord(record.backend) ? record.backend : null;
  const imageDigests = record && isRecord(record.image_digests) ? record.image_digests : null;
  const storage = record && isRecord(record.storage) ? record.storage : null;
  const clipStore = storage && isRecord(storage.clip_store) ? storage.clip_store : null;
  if (
    !record
    || !backend
    || typeof backend.configured !== 'boolean'
    || !('reachable' in backend) || !isNullableBoolean(backend.reachable)
    || !('last_ok_at' in backend) || !isNullableString(backend.last_ok_at)
    || typeof record.version !== 'string'
    || !imageDigests
    || !('ml_api' in imageDigests) || !isNullableString(imageDigests.ml_api)
    || !('ml_worker' in imageDigests) || !isNullableString(imageDigests.ml_worker)
    || !clipStore
    || !('total_bytes' in clipStore) || !isNullableInteger(clipStore.total_bytes)
    || !('used_bytes' in clipStore) || !isNullableInteger(clipStore.used_bytes)
    || !('used_pct' in clipStore) || !isNullableFiniteNumber(clipStore.used_pct)
    || typeof record.updated_at !== 'string'
  ) {
    throw new Error('Invalid system response');
  }

  return {
    updated_at: record.updated_at,
    backend: {
      configured: backend.configured,
      reachable: backend.reachable,
      last_ok_at: backend.last_ok_at,
    },
    version: normalizeVersion(record),
    image_digests: {
      ml_api: imageDigests.ml_api,
      ml_worker: imageDigests.ml_worker,
    },
    storage: {
      clip_store: {
        total_bytes: clipStore.total_bytes,
        used_bytes: clipStore.used_bytes,
        used_pct: clipStore.used_pct,
      },
    },
  };
}
