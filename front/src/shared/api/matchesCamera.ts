import type { Camera } from '@/shared/api/types';

export function matchesCamera(camera: Camera, id: string): boolean {
  return camera.id === id || (!!camera.backend_camera_id && camera.backend_camera_id === id);
}
