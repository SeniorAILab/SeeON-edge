import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Camera, StatusSnapshot } from '@/shared/api/client';
import type { DashboardLocation } from '@/app/dashboardLocation';
import { OperationsView } from '@/features/operations/OperationsView';

const cleanups: Array<() => void> = [];
afterEach(() => { while (cleanups.length) cleanups.pop()?.(); });

function camera(id: string, backend = id): Camera {
  return { id, backend_camera_id: backend, label: `카메라 ${id}`, floor_name: '1층', space_id: 'room-1', space_name: '101호', rtsp_url_masked: '', status: 'unknown', created_at: null };
}

function renderView(options: { cameras?: Camera[]; location?: DashboardLocation; viewport?: 'desktop' | 'mobile'; status?: StatusSnapshot | null } = {}) {
  const host = document.createElement('div');
  host.id = 'main-content';
  document.body.append(host);
  const root: Root = createRoot(host);
  const navigate = vi.fn();
  const replace = vi.fn();
  const cameras = options.cameras ?? [camera('local', 'worker/cam')];
  const location = options.location ?? { page: 'operations', mode: 'wall', wallPage: '1' };
  act(() => root.render(<OperationsView active cameras={cameras} camerasState="success" status={options.status ?? null} statusState="success" location={location} onNavigate={navigate} onReplace={replace} viewport={options.viewport ?? 'desktop'} />));
  cleanups.push(() => { act(() => root.unmount()); host.remove(); });
  return { host, navigate, replace, rerender: (next: DashboardLocation, nextCameras = cameras) => act(() => root.render(<OperationsView active cameras={nextCameras} camerasState="success" status={options.status ?? null} statusState="success" location={next} onNavigate={navigate} onReplace={replace} viewport={options.viewport ?? 'desktop'} />)) };
}

describe('OperationsView', () => {
  it('renders honest empty/loading/error states and retry actions', () => {
    const host = document.createElement('div'); document.body.append(host); const root = createRoot(host); const retry = vi.fn();
    act(() => root.render(<OperationsView active cameras={[]} camerasState="loading" status={null} statusState="loading" location={{ page: 'operations', mode: 'wall', wallPage: '1' }} onNavigate={vi.fn()} onReplace={vi.fn()} viewport="desktop" onRetryCameras={retry} />));
    expect(host.textContent).toContain('카메라 불러오는 중');
    expect(host.querySelector('[data-camera-refresh-status]')).toBeNull();
    act(() => root.render(<OperationsView active cameras={[]} camerasState="error" status={null} statusState="error" location={{ page: 'operations', mode: 'wall', wallPage: '1' }} onNavigate={vi.fn()} onReplace={vi.fn()} viewport="desktop" onRetryCameras={retry} />));
    expect(host.textContent).toContain('카메라 목록을 불러올 수 없음');
    expect(host.querySelector('[data-camera-refresh-status]')).toBeNull();
    act(() => Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '다시 시도')?.click());
    expect(retry).toHaveBeenCalled();
    act(() => root.render(<OperationsView active cameras={[]} camerasState="success" status={null} statusState="success" location={{ page: 'operations', mode: 'wall', wallPage: '1' }} onNavigate={vi.fn()} onReplace={vi.fn()} viewport="desktop" />));
    expect(host.textContent).toContain('등록된 카메라가 없습니다');
    act(() => root.unmount()); host.remove();
  });

  it('offers a semantic camera-management action from the successful empty state', () => {
    const { host, navigate } = renderView({ cameras: [] });
    const action = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent === '카메라 등록하기');

    expect(action?.classList.contains('brand-action')).toBe(true);
    act(() => action?.click());
    expect(navigate).toHaveBeenCalledWith({
      page: 'cameras', floor: null, room: null, camera: null, mode: null,
      event: null, clip: null, wallPage: null,
    });
  });

  it('shows retained camera data with its real last-success time and clears freshness copy after recovery', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    const lastSuccessAt = Date.parse('2026-07-21T00:15:00Z');
    const common = {
      active: true,
      status: null,
      statusState: 'success' as const,
      location: { page: 'operations', mode: 'wall', wallPage: '1' } as DashboardLocation,
      onNavigate: vi.fn(),
      onReplace: vi.fn(),
      viewport: 'desktop' as const,
    };

    act(() => root.render(<OperationsView {...common} cameras={[camera('local')]} camerasState="error" camerasLastSuccessAt={lastSuccessAt} />));
    expect(host.textContent).toContain('마지막 카메라 목록을 표시합니다.');
    expect(host.textContent).toContain(`마지막 확인 ${new Date(lastSuccessAt).toLocaleString('ko-KR')}`);

    act(() => root.render(<OperationsView {...common} cameras={[camera('local')]} camerasState="success" camerasLastSuccessAt={Date.parse('2026-07-21T00:20:00Z')} />));
    expect(host.textContent).not.toContain('마지막 카메라 목록을 표시합니다.');
    expect(host.textContent).not.toContain('마지막 확인');

    act(() => root.render(<OperationsView {...common} cameras={[]} camerasState="error" camerasLastSuccessAt={null} />));
    expect(host.textContent).toContain('카메라 목록을 불러올 수 없음');
    expect(host.textContent).not.toContain('마지막 확인');
    act(() => root.unmount());
    host.remove();
  });

  it('syncs filters/page and exposes accessible selected-card state without stream URLs in wall', () => {
    const cameras = Array.from({ length: 13 }, (_, index) => ({ ...camera(index === 0 ? 'local' : String(index), index === 0 ? 'worker/cam' : String(index)), label: index === 0 ? '카메라 local' : `카메라 ${String(index).padStart(2, '0')}`, floor_name: index === 12 ? '2층' : '1층' }));
    const { host, navigate } = renderView({ cameras });
    act(() => Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '다음')?.click());
    expect(navigate).toHaveBeenCalledWith({ wallPage: '2' });
    const selects = host.querySelectorAll('select');
    act(() => { (selects[0] as HTMLSelectElement).value = '1층'; selects[0].dispatchEvent(new Event('change', { bubbles: true })); });
    expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ floor: '1층', wallPage: '1' }));
    const card = host.querySelector<HTMLButtonElement>('[aria-label="카메라 local 선택"]');
    expect(card?.getAttribute('aria-pressed')).toBe('false');
    act(() => card?.click());
    expect(navigate).toHaveBeenCalledWith({ camera: 'local', mode: 'wall' });
    expect(host.querySelector('[data-stream]')).toBeNull();
    expect(host.innerHTML).not.toContain('/streams/worker%2Fcam?');
  });

  it('uses one responsive camera grid per floor across distinct rooms', () => {
    const cameras = [
      { ...camera('c'), space_id: 'room-c', space_name: '다실', label: '다 카메라' },
      { ...camera('a'), space_id: 'room-a', space_name: '가실', label: '가 카메라' },
      { ...camera('b'), space_id: 'room-b', space_name: '나실', label: '나 카메라' },
    ];
    const { host } = renderView({ cameras });
    const floor = host.querySelector('section[aria-labelledby="floor-1층"]');
    const floorGrid = Array.from(floor?.children ?? []).find((element) => element.tagName === 'DIV' && element.classList.contains('grid'));

    expect(floorGrid?.classList.contains('md:grid-cols-2')).toBe(true);
    expect(floorGrid?.classList.contains('xl:grid-cols-3')).toBe(true);
    expect(floorGrid?.querySelectorAll(':scope > article')).toHaveLength(3);
    expect(Array.from(floorGrid?.querySelectorAll(':scope > article > h3') ?? []).map((heading) => heading.textContent)).toEqual(['가실', '나실', '다실']);
    expect(Array.from(floorGrid?.querySelectorAll(':scope > article button[aria-label]') ?? []).map((card) => card.getAttribute('aria-label'))).toEqual(['가 카메라 선택', '나 카메라 선택', '다 카메라 선택']);
  });

  it('uses local query identity and backend media identity for snapshot and focus', () => {
    const { host, navigate, rerender } = renderView();
    expect(host.querySelector('img')?.getAttribute('src')).toContain('/streams/worker%2Fcam/snapshot');
    act(() => host.querySelector<HTMLButtonElement>('[aria-label="카메라 local 선택"]')?.click());
    expect(navigate).toHaveBeenCalledWith({ camera: 'local', mode: 'wall' });
    rerender({ page: 'operations', camera: 'local', mode: 'wall', wallPage: '1' });
    act(() => Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '집중 보기')?.click());
    expect(navigate).toHaveBeenLastCalledWith({ camera: 'local', mode: 'focus' });
    rerender({ page: 'operations', camera: 'local', mode: 'focus', wallPage: '1' });
    expect(host.querySelectorAll('[data-stream]')).toHaveLength(1);
    expect(host.querySelector('[data-stream]')?.getAttribute('src')).toContain('/streams/worker%2Fcam');
  });

  it('unmounts the single focus stream on back, page change, and camera switch and reports focus failure', () => {
    const cameras = [camera('a', 'worker-a'), camera('b', 'worker-b')];
    const { host, rerender } = renderView({ cameras, location: { page: 'operations', camera: 'a', mode: 'focus', wallPage: '1' } });
    const first = host.querySelector('[data-stream]');
    expect(first).not.toBeNull();
    act(() => first?.dispatchEvent(new Event('error', { bubbles: true })));
    expect(host.textContent).toContain('라이브 영상을 불러올 수 없음');
    rerender({ page: 'operations', camera: 'b', mode: 'focus', wallPage: '1' });
    expect(first?.isConnected).toBe(false);
    expect(host.querySelectorAll('[data-stream]')).toHaveLength(1);
    rerender({ page: 'operations', camera: 'b', mode: 'wall', wallPage: '1' });
    expect(host.querySelector('[data-stream]')).toBeNull();
    rerender({ page: 'events' });
    expect(host.querySelector('[data-stream]')).toBeNull();
  });

  it('retries a failed focus stream only when its resolved media identity changes', () => {
    const location: DashboardLocation = { page: 'operations', camera: 'local', mode: 'focus', wallPage: '1' };
    const { host, rerender } = renderView({ cameras: [camera('local', 'worker-old')], location });
    const failedStream = host.querySelector('[data-stream]');

    act(() => failedStream?.dispatchEvent(new Event('error', { bubbles: true })));
    expect(host.querySelector('[data-stream]')).toBeNull();
    expect(host.textContent).toContain('라이브 영상을 불러올 수 없음');

    rerender(location, [camera('local', 'worker-old')]);
    expect(host.querySelector('[data-stream]')).toBeNull();

    rerender(location, [camera('local', 'worker-new')]);
    const recoveredStream = host.querySelector('[data-stream]');
    expect(host.querySelectorAll('[data-stream]')).toHaveLength(1);
    expect(recoveredStream?.getAttribute('src')).toContain('/streams/worker-new');
    act(() => recoveredStream?.dispatchEvent(new Event('load', { bubbles: true })));
    expect(host.querySelector('[data-stream]')).toBe(recoveredStream);
    expect(host.textContent).not.toContain('라이브 영상을 불러올 수 없음');
  });

  it('reports the focused stream lifecycle and resets it for a new media identity', () => {
    const location: DashboardLocation = { page: 'operations', camera: 'local', mode: 'focus', wallPage: '1' };
    const { host, rerender } = renderView({ cameras: [camera('local', 'worker-old')], location });
    const oldStream = host.querySelector('[data-stream]');
    const liveStatus = host.querySelector('[role="status"]');
    expect(host.querySelectorAll('[role="status"]')).toHaveLength(1);
    expect(liveStatus?.getAttribute('aria-live')).toBe('polite');
    expect(liveStatus?.getAttribute('aria-atomic')).toBe('true');
    expect(liveStatus?.textContent).toBe('라이브 영상 불러오는 중');

    act(() => oldStream?.dispatchEvent(new Event('load', { bubbles: true })));
    expect(host.querySelector('[role="status"]')).toBe(liveStatus);
    expect(liveStatus?.textContent).toBe('라이브 영상 연결됨');

    rerender(location, [camera('local', 'worker-new')]);
    const newStream = host.querySelector('[data-stream]');
    const newLiveStatus = host.querySelector('[role="status"]');
    expect(newStream).not.toBe(oldStream);
    expect(newLiveStatus).not.toBe(liveStatus);
    expect(host.querySelectorAll('[role="status"]')).toHaveLength(1);
    expect(newLiveStatus?.textContent).toBe('라이브 영상 불러오는 중');

    act(() => newStream?.dispatchEvent(new Event('error', { bubbles: true })));
    expect(host.querySelectorAll('[role="status"]')).toHaveLength(1);
    expect(host.querySelector('[role="status"]')).toBe(newLiveStatus);
    expect(newLiveStatus?.textContent).toBe('라이브 영상을 불러올 수 없음');
    expect(host.textContent?.match(/라이브 영상을 불러올 수 없음/g)).toHaveLength(1);

    rerender(location, [camera('local', 'worker-recovered')]);
    const recoveredLiveStatus = host.querySelector('[role="status"]');
    expect(recoveredLiveStatus).not.toBe(newLiveStatus);
    expect(recoveredLiveStatus?.textContent).toBe('라이브 영상 불러오는 중');
    act(() => host.querySelector('[data-stream]')?.dispatchEvent(new Event('load', { bubbles: true })));
    expect(host.querySelector('[role="status"]')).toBe(recoveredLiveStatus);
    expect(recoveredLiveStatus?.textContent).toBe('라이브 영상 연결됨');
  });

  it('does not show an old snapshot when its late load arrives after a media identity change', () => {
    const location: DashboardLocation = { page: 'operations', mode: 'wall', wallPage: '1' };
    const { host, rerender } = renderView({ cameras: [camera('local', 'worker-old')], location });
    const oldRequest = host.querySelector<HTMLImageElement>('[data-snapshot-preload]');

    rerender(location, [camera('local', 'worker-new')]);
    const newRequest = host.querySelector<HTMLImageElement>('[data-snapshot-preload]');
    expect(newRequest?.getAttribute('src')).toContain('/streams/worker-new/snapshot');
    act(() => newRequest?.dispatchEvent(new Event('error', { bubbles: true })));
    act(() => oldRequest?.dispatchEvent(new Event('load', { bubbles: true })));

    expect(host.textContent).toContain('영상을 불러올 수 없음');
    expect(host.querySelector('img:not([data-snapshot-preload])')).toBeNull();
    expect(host.innerHTML).not.toContain('/streams/worker-old/snapshot');
  });

  it('shows snapshot failure separately from liveness and never paints a loaded-success state early', () => {
    const heartbeat = { camera_id: 'local', facility_id: null, status: 'online' as const, last_heartbeat_at: null, age_sec: 1, config_version: null };
    const status: StatusSnapshot = { cameras: { local: heartbeat }, stale_after_sec: null, runtime: { facilities: {}, stale_after_sec: null } };
    const { host } = renderView({ status });
    expect(host.textContent).toContain('연결됨');
    expect(host.textContent).toContain('영상 불러오는 중');
    const image = host.querySelector('img');
    act(() => image?.dispatchEvent(new Event('error', { bubbles: true })));
    expect(host.textContent).toContain('영상을 불러올 수 없음');
    expect(host.textContent).toContain('연결됨');
  });

  it('uses semantic action and media-overlay classes instead of raw color aliases', () => {
    const { host, rerender } = renderView({ location: { page: 'operations', camera: 'local', mode: 'wall', wallPage: '1' } });

    expect(host.querySelector('.brand-action')?.textContent).toBe('집중 보기');
    expect(host.querySelector('.media-status-overlay')?.textContent).toContain('영상 불러오는 중');
    rerender({ page: 'operations', camera: 'local', mode: 'focus', wallPage: '1' });
    expect(host.querySelector('.media-status-overlay')?.textContent).toContain('라이브 영상 불러오는 중');
  });

  it('keeps the last-good snapshot timestamp after a successful image load', () => {
    const { host } = renderView();
    const candidate = host.querySelector<HTMLImageElement>('[data-snapshot-preload]');
    act(() => candidate?.dispatchEvent(new Event('load', { bubbles: true })));
    expect(host.textContent).toContain('영상 확인됨');
    expect(host.textContent).toContain('마지막 영상');
    expect(host.querySelector('img:not([data-snapshot-preload])')).toBe(candidate);
  });

  it('keeps the last-good frame visible until a refreshed snapshot loads', () => {
    vi.useFakeTimers();
    const { host } = renderView();
    const firstRequest = host.querySelector('img')?.getAttribute('src');
    act(() => host.querySelector('img')?.dispatchEvent(new Event('load', { bubbles: true })));
    const lastGood = host.querySelector<HTMLImageElement>('img:not([data-snapshot-preload])');

    act(() => vi.advanceTimersByTime(6_000));

    expect(host.querySelector('img:not([data-snapshot-preload])')).toBe(lastGood);
    expect(lastGood?.getAttribute('src')).toBe(firstRequest);
    const failedRefresh = host.querySelector<HTMLImageElement>('[data-snapshot-preload]');
    expect(failedRefresh?.getAttribute('src')).not.toBe(firstRequest);
    act(() => host.querySelector('[data-snapshot-preload]')?.dispatchEvent(new Event('error', { bubbles: true })));
    expect(host.querySelector('img:not([data-snapshot-preload])')).toBe(lastGood);
    expect(failedRefresh?.isConnected).toBe(false);

    act(() => vi.advanceTimersByTime(6_000));
    const successfulRefresh = host.querySelector<HTMLImageElement>('[data-snapshot-preload]');
    act(() => successfulRefresh?.dispatchEvent(new Event('load', { bubbles: true })));
    expect(host.querySelector('img:not([data-snapshot-preload])')).toBe(successfulRefresh);
    expect(lastGood?.isConnected).toBe(false);
  });

  it('labels stale and never-seen liveness without changing the image lifecycle', () => {
    const cameras = [camera('stale'), camera('never')];
    const status: StatusSnapshot = {
      cameras: {
        stale: { camera_id: 'stale', facility_id: null, status: 'stale', last_heartbeat_at: null, age_sec: 90, config_version: null },
        never: { camera_id: 'never', facility_id: null, status: 'never_seen', last_heartbeat_at: null, age_sec: null, config_version: null },
      },
      stale_after_sec: null,
      runtime: { facilities: {}, stale_after_sec: null },
    };
    const { host } = renderView({ cameras, status });
    expect(host.textContent).toContain('연결 지연');
    expect(host.textContent).toContain('연결 이력 없음');
    expect(host.textContent?.match(/영상 불러오는 중/g)).toHaveLength(2);
  });

  it('uses a desktop complementary inspector and mobile accessible bottom sheet', () => {
    const desktop = renderView({ location: { page: 'operations', camera: 'local', mode: 'wall', wallPage: '1' }, viewport: 'desktop' });
    expect(desktop.host.querySelector('[role="complementary"]')?.getAttribute('aria-label')).toContain('카메라 local');
    const mobile = renderView({ location: { page: 'operations', camera: 'local', mode: 'wall', wallPage: '1' }, viewport: 'mobile' });
    expect(document.body.querySelector('[role="dialog"][aria-modal="true"]')).not.toBeNull();
  });

  it.each([375, 768])('focuses the mobile selection-clear control at %ipx and restores the invoking card on close', (width) => {
    const previousWidth = window.innerWidth;
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: width });
    cleanups.push(() => Object.defineProperty(window, 'innerWidth', { configurable: true, value: previousWidth }));
    const { host, navigate, rerender } = renderView({ viewport: 'mobile' });
    host.id = 'main-content';
    const card = host.querySelector<HTMLButtonElement>('[aria-label="카메라 local 선택"]');
    card?.focus();
    act(() => card?.click());
    rerender({ page: 'operations', camera: 'local', mode: 'wall', wallPage: '1' });

    const closeSelection = Array.from(document.body.querySelectorAll<HTMLButtonElement>('[role="dialog"] button'))
      .find((button) => button.textContent === '선택 해제');
    expect(document.activeElement).toBe(closeSelection);
    act(() => closeSelection?.click());
    expect(navigate).toHaveBeenLastCalledWith({ camera: null, mode: 'wall' });
    rerender({ page: 'operations', mode: 'wall', wallPage: '1' });

    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
    expect(document.activeElement).toBe(card);
  });

  it('restores the operations heading fallback when a direct-linked mobile sheet closes without an invoker', () => {
    const { host, navigate, rerender } = renderView({
      viewport: 'mobile',
      location: { page: 'operations', camera: 'local', mode: 'wall', wallPage: '1' },
    });
    const clearSelection = Array.from(document.body.querySelectorAll<HTMLButtonElement>('[role="dialog"] button'))
      .find((button) => button.textContent === '선택 해제');
    expect(document.activeElement).toBe(clearSelection);

    act(() => clearSelection?.click());
    expect(navigate).toHaveBeenLastCalledWith({ camera: null, mode: 'wall' });
    rerender({ page: 'operations', mode: 'wall', wallPage: '1' });

    expect(document.activeElement).toBe(host.querySelector('#operations-title'));
  });

  it('focuses the destination heading when mobile inspector navigation removes its invoker', () => {
    const { host, rerender } = renderView({ viewport: 'mobile' });
    host.id = 'main-content';
    const card = host.querySelector<HTMLButtonElement>('[aria-label="카메라 local 선택"]');
    card?.focus();
    act(() => card?.click());
    rerender({ page: 'operations', camera: 'local', mode: 'wall', wallPage: '1' });

    const focusButton = Array.from(document.body.querySelectorAll<HTMLButtonElement>('[role="dialog"] button'))
      .find((button) => button.textContent === '집중 보기');
    focusButton?.focus();
    act(() => focusButton?.click());
    rerender({ page: 'operations', camera: 'local', mode: 'focus', wallPage: '1' });

    const heading = host.querySelector('#focused-camera-title');
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
    expect(document.activeElement).toBe(heading);
    expect(document.activeElement).not.toBe(document.body);
  });

  it('announces selection invalidation and corrects page without selecting another camera', () => {
    const { host, replace } = renderView({ cameras: Array.from({ length: 13 }, (_, index) => camera(String(index))), location: { page: 'operations', camera: 'gone', mode: 'focus', wallPage: '9' } });
    expect(replace).toHaveBeenCalledWith({ camera: null, mode: 'wall', wallPage: '2' });
    expect(host.querySelector('[aria-live="polite"]')?.textContent).toContain('선택한 카메라');
    expect(host.querySelector('[aria-pressed="true"]')).toBeNull();
  });
});
