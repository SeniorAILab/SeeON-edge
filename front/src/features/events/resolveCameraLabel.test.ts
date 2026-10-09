import { describe, expect, it } from 'vitest';
import { resolveCameraLabel } from '@/features/events/resolveCameraLabel';
import type { Camera, Clip } from '@/shared/api/types';

const cameras = [
  { id: 'local-1', backend_camera_id: 'hub-1', label: '서울 301호' },
  { id: 'local-2', backend_camera_id: null, label: '서울 302호' },
] as Camera[];

const clip = (camera_id: string) => ({ camera_id, camera_label: 'fallback' }) as Clip;

describe('resolveCameraLabel', () => {
  it('matches a clip stamped with the local id', () => {
    expect(resolveCameraLabel(cameras, clip('local-2'))).toBe('서울 302호');
  });

  it('matches a clip stamped with backend_camera_id', () => {
    expect(resolveCameraLabel(cameras, clip('hub-1'))).toBe('서울 301호');
  });

  it('falls back to clip.camera_label for an unknown id', () => {
    expect(resolveCameraLabel(cameras, clip('nope'))).toBe('fallback');
  });
});
