import { describe, expect, it } from 'vitest';
import type { Camera, Clip } from '@/shared/api/client';
import {
  eventLabel,
  eventHistoryLocationUpdate,
  filterHistoricalClips,
  groupHistoricalClips,
  normalizeEventTypeName,
  resolveClipCamera,
  sortHistoricalClips,
} from '@/features/events/cameraEventLogic';
import { canonicalizeDashboardLocation, serializeDashboardLocation } from '@/app/dashboardLocation';

const camera = (id: string, floor: string | null, room: string | null, alias: string | null = null): Camera => ({
  id,
  label: id,
  rtsp_url_masked: 'rtsp://***',
  floor_name: floor,
  space_id: room,
  space_name: room,
  backend_camera_id: alias,
  status: 'online',
  created_at: null,
});

const clip = (id: string, cameraId: string | null, eventType: string, createdAt: string | null): Clip => ({
  id,
  camera_id: cameraId,
  camera_label: cameraId ?? '카메라 미상',
  event_type: eventType,
  created_at: createdAt,
  label: null,
  reviewState: 'unknown',
  video_path: `/api/v1/clips/${id}/video`,
  video_available: true,
  video_error: null,
});

describe('historical clip logic', () => {
  const cameras = [
    camera('local-a', '2층', '공용실', 'worker-a'),
    camera('local-b', '3층', '공용실'),
  ];
  const clips = [
    clip('old', 'local-a', 'fall', '2026-07-19T01:00:00Z'),
    clip('new', 'worker-a', 'bed-exit', '2026-07-20T01:00:00Z'),
    clip('same', 'local-b', 'fall', '2026-07-20T01:00:00Z'),
    clip('invalid', 'local-a', 'fall', 'not-a-time'),
    clip('missing', 'local-a', 'fall', null),
    clip('orphan', 'gone', 'fall', '2026-07-21T01:00:00Z'),
  ];

  it('resolves a unique backend alias and keeps ambiguous aliases unresolved', () => {
    expect(resolveClipCamera(clips[1], cameras)?.id).toBe('local-a');
    expect(resolveClipCamera(clips[1], [...cameras, camera('local-c', null, null, 'worker-a')])).toBeNull();
  });

  it('sorts valid timestamps newest-first, stably, with invalid and missing timestamps last', () => {
    expect(sortHistoricalClips(clips).map((entry) => entry.id)).toEqual([
      'orphan', 'new', 'same', 'old', 'invalid', 'missing',
    ]);
  });

  it('filters exact event types and separates duplicate room names by floor', () => {
    expect(filterHistoricalClips(cameras, clips, { floor: '2층', room: '공용실', event: 'bed-exit' }).map((entry) => entry.id)).toEqual(['new']);
    expect(filterHistoricalClips(cameras, clips, { floor: '3층', room: '공용실' }).map((entry) => entry.id)).toEqual(['same']);
    expect(filterHistoricalClips(cameras, clips, { event: 'fall' }).map((entry) => entry.id)).toEqual(['orphan', 'same', 'old', 'invalid', 'missing']);
    expect(groupHistoricalClips(cameras, clips).map((group) => `${group.floor}/${group.room}`)).toEqual([
      '2층/공용실', '3층/공용실', '층 정보 없음/공간 정보 없음',
    ]);
  });

  it('shows orphan clips only without location filters', () => {
    expect(filterHistoricalClips(cameras, [clips[5]], {}).map((entry) => entry.id)).toEqual(['orphan']);
    expect(filterHistoricalClips(cameras, [clips[5]], { camera: 'local-a' })).toEqual([]);
  });

  it('clears dependent URL state in floor-room-camera-event-clip order', () => {
    const full = { page: 'events' as const, floor: '2층', room: '공용실', camera: 'local-a', event: 'fall', clip: 'old' };
    expect(eventHistoryLocationUpdate(full, 'floor', '3층')).toEqual({ page: 'events', floor: '3층' });
    expect(eventHistoryLocationUpdate(full, 'room', null)).toEqual({ page: 'events', floor: '2층' });
    expect(eventHistoryLocationUpdate(full, 'camera', 'local-b')).toEqual({ page: 'events', floor: '2층', room: '공용실', camera: 'local-b' });
    expect(eventHistoryLocationUpdate(full, 'event', 'bed-exit')).toEqual({ page: 'events', floor: '2층', room: '공용실', camera: 'local-a', event: 'bed-exit' });
    expect(eventHistoryLocationUpdate(full, 'clip', null)).toEqual({ page: 'events', floor: '2층', room: '공용실', camera: 'local-a', event: 'fall' });
  });

  it('round-trips exact historical filters and clip selection through canonical URL state', () => {
    const location = { page: 'events' as const, floor: '2층', room: '공용실', camera: 'local-a', event: 'bed-exit', clip: 'new' };
    const search = serializeDashboardLocation(location);
    expect(search).toContain('%EC%B8%B5');
    expect(canonicalizeDashboardLocation(search, {
      cameras: { status: 'success', data: cameras },
      clips: { status: 'success', data: clips },
    }).location).toEqual(location);
  });

  it.each([
    ['fall', 'fall', '낙상'],
    [' FALL ', 'fall', '낙상'],
    ['bed-exit', 'bed-exit', '침대 이탈'],
    ['BED_EXIT', 'bed-exit', '침대 이탈'],
    ['bed exit', 'bed-exit', '침대 이탈'],
  ])('labels the exact supported normalized event type %j', (eventType, normalized, label) => {
    expect(normalizeEventTypeName(eventType)).toBe(normalized);
    expect(eventLabel(eventType)).toBe(label);
  });

  it.each([
    'not_fall',
    'fall-risk',
    'night-bed-exit',
    'bed-exit-warning',
    '낙상',
    '침대 이탈',
  ])('preserves the truthful name for unsupported event type %j', (eventType) => {
    expect(eventLabel(eventType)).toBe(eventType);
  });

  it('uses the generic normalized identifier for missing event types without inventing a supported label', () => {
    expect(normalizeEventTypeName(null)).toBe('event');
    expect(normalizeEventTypeName(undefined)).toBe('event');
    expect(normalizeEventTypeName('')).toBe('event');
    expect(eventLabel('')).toBe('');
  });
});
