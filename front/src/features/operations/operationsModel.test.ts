import { describe, expect, it } from 'vitest';
import type { Camera, StatusSnapshot } from '@/shared/api/client';
import { filterAndGroupCameras, getCameraLiveness, paginateCameras, snapshotJitterMs } from '@/features/operations/operationsModel';

function camera(id: string, floor: string | null, room: string | null, label: string, roomName = room): Camera {
  return { id, label, floor_name: floor, space_id: room, space_name: roomName, backend_camera_id: null, rtsp_url_masked: '', status: 'unknown', created_at: null };
}

describe('operations camera model', () => {
  it('sorts and groups floor → room → label with Korean collation, separate duplicate rooms, and 미분류', () => {
    const cameras = [
      camera('5', null, null, '미지정'),
      camera('4', '2층', 'room-a', '나 카메라', '공용실'),
      camera('3', '1층', 'room-b', '가 카메라', '공용실'),
      camera('2', '1층', 'room-b', '나 카메라', '공용실'),
      camera('1', '1층', null, '복도'),
    ];

    expect(filterAndGroupCameras(cameras, {}).groups.map((floor) => [
      floor.label,
      floor.rooms.map((room) => [room.id, room.label, room.cameras.map((entry) => entry.id)]),
    ])).toEqual([
      ['1층', [['room-b', '공용실', ['3', '2']], ['미분류', '미분류', ['1']]]],
      ['2층', [['room-a', '공용실', ['4']]]],
      ['미분류', [['미분류', '미분류', ['5']]]],
    ]);
  });

  it('filters exact floor and room values', () => {
    const cameras = [camera('1', '1층', 'a', 'A'), camera('2', '2층', 'a', 'B'), camera('3', '1층', 'b', 'C')];
    expect(filterAndGroupCameras(cameras, { floor: '1층', room: 'a' }).cameras.map((entry) => entry.id)).toEqual(['1']);
  });

  it.each([[0, 1, 0], [1, 1, 1], [12, 1, 12], [13, 2, 1], [50, 5, 2]])(
    'paginates %i cameras deterministically',
    (count, requestedPage, expectedLength) => {
      const cameras = Array.from({ length: count }, (_, index) => camera(String(index), '1층', 'r', String(index).padStart(2, '0')));
      const page = paginateCameras(cameras, requestedPage);
      expect(page.pageCount).toBe(Math.max(1, Math.ceil(count / 12)));
      expect(page.page).toBe(Math.min(requestedPage, page.pageCount));
      expect(page.cameras).toHaveLength(expectedLength);
    },
  );

  it('clamps an oversized page for 50 cameras and uses stable deterministic jitter', () => {
    const cameras = Array.from({ length: 50 }, (_, index) => camera(String(index), '1층', 'r', String(index).padStart(2, '0')));
    expect(paginateCameras(cameras, 99)).toMatchObject({ page: 5, pageCount: 5 });
    expect(snapshotJitterMs('camera-a')).toBe(snapshotJitterMs('camera-a'));
    expect(snapshotJitterMs('camera-a')).toBeGreaterThanOrEqual(0);
    expect(snapshotJitterMs('camera-a')).toBeLessThan(1_000);
  });

  it('keeps liveness separate and rejects ambiguous aliases', () => {
    const first = { ...camera('local-a', '1층', 'r', 'A'), backend_camera_id: 'shared' };
    const second = { ...camera('local-b', '1층', 'r', 'B'), backend_camera_id: 'shared' };
    const status: StatusSnapshot = {
      cameras: { shared: { camera_id: 'shared', facility_id: null, status: 'online', last_heartbeat_at: null, age_sec: 1, config_version: null } },
      stale_after_sec: null,
      runtime: { facilities: {}, stale_after_sec: null },
    };
    expect(getCameraLiveness(first, [first, second], status)).toBe('unknown');
    expect(getCameraLiveness(first, [first], status)).toBe('online');
    expect(getCameraLiveness(first, [first], null)).toBe('unknown');
  });
});
