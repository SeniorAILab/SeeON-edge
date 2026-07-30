import type { Camera, Clip } from '@/shared/api/client';

export type EventHistoryLocation = {
  page: 'events';
  floor?: string;
  room?: string;
  camera?: string;
  event?: string;
  clip?: string;
};

export function formatEventTime(value: string | null): string {
  if (!value) return '시간 정보 없음';
  const time = Date.parse(value);
  return Number.isFinite(time) ? new Date(time).toLocaleString('ko-KR') : '시간 정보 없음';
}

export function resolveClipCamera(clip: Clip, cameras: readonly Camera[]): Camera | null {
  if (!clip.camera_id) return null;
  const local = cameras.find((camera) => camera.id === clip.camera_id);
  if (local) return local;
  const aliases = cameras.filter((camera) => camera.backend_camera_id === clip.camera_id);
  return aliases.length === 1 ? aliases[0] : null;
}

export function sortHistoricalClips(clips: readonly Clip[]): Clip[] {
  return clips
    .map((clip, index) => ({ clip, index, time: clip.created_at ? Date.parse(clip.created_at) : Number.NaN }))
    .sort((left, right) => {
      const leftValid = Number.isFinite(left.time);
      const rightValid = Number.isFinite(right.time);
      if (leftValid !== rightValid) return leftValid ? -1 : 1;
      if (leftValid && left.time !== right.time) return right.time - left.time;
      return left.index - right.index;
    })
    .map(({ clip }) => clip);
}

export function filterHistoricalClips(
  cameras: readonly Camera[],
  clips: readonly Clip[],
  filters: Omit<EventHistoryLocation, 'page' | 'clip'>,
): Clip[] {
  return sortHistoricalClips(clips.filter((clip) => {
    const camera = resolveClipCamera(clip, cameras);
    if (!camera) return !filters.floor && !filters.room && !filters.camera && (!filters.event || clip.event_type === filters.event);
    return (!filters.floor || camera.floor_name === filters.floor)
      && (!filters.room || camera.space_id === filters.room)
      && (!filters.camera || camera.id === filters.camera)
      && (!filters.event || clip.event_type === filters.event);
  }));
}

export function groupHistoricalClips(cameras: readonly Camera[], clips: readonly Clip[]): Array<{
  key: string;
  floor: string;
  room: string;
  clips: Clip[];
}> {
  const groups = new Map<string, { key: string; floor: string; room: string; clips: Clip[] }>();
  clips.forEach((clip) => {
    const camera = resolveClipCamera(clip, cameras);
    const floor = camera?.floor_name ?? '층 정보 없음';
    const room = camera?.space_name ?? camera?.space_id ?? '공간 정보 없음';
    const key = JSON.stringify([camera?.floor_name ?? null, camera?.space_id ?? null]);
    const group = groups.get(key) ?? { key, floor, room, clips: [] };
    group.clips.push(clip);
    groups.set(key, group);
  });
  return [...groups.values()].sort((left, right) => {
    const leftMissing = left.floor === '층 정보 없음';
    const rightMissing = right.floor === '층 정보 없음';
    if (leftMissing !== rightMissing) return leftMissing ? 1 : -1;
    return left.floor.localeCompare(right.floor, 'ko') || left.room.localeCompare(right.room, 'ko');
  });
}

export function eventHistoryLocationUpdate(
  location: EventHistoryLocation,
  key: 'floor' | 'room' | 'camera' | 'event' | 'clip',
  value: string | null,
): EventHistoryLocation {
  const next = { ...location };
  const dependentKeys: Record<typeof key, Array<keyof EventHistoryLocation>> = {
    floor: ['room', 'camera', 'event', 'clip'],
    room: ['camera', 'event', 'clip'],
    camera: ['event', 'clip'],
    event: ['clip'],
    clip: [],
  };
  if (value) next[key] = value;
  else delete next[key];
  dependentKeys[key].forEach((dependent) => delete next[dependent]);
  return next;
}

export function normalizeEventTypeName(value: string | null | undefined): string {
  if (!value) return 'event';
  const normalized = value.trim().toLowerCase().replace(/[\s_]+/g, '-');
  return normalized || 'event';
}

export function eventLabel(type: string): string {
  const normalized = normalizeEventTypeName(type);
  if (normalized === 'bed-exit') return '침대 이탈';
  if (normalized === 'fall') return '낙상';
  return type;
}
