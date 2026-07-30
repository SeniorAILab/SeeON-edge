import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Camera, Clip } from '@/shared/api/client';
import { EventHistoryPage } from '@/features/events/EventHistoryPage';
import type { EventHistoryLocation } from '@/features/events/cameraEventLogic';

const cameras: Camera[] = [
  { id: 'local-a', backend_camera_id: 'worker-a', label: '동쪽 카메라', rtsp_url_masked: 'rtsp://***', floor_name: '2층', space_id: 'shared', space_name: '공용실', status: 'online', created_at: null },
  { id: 'local-b', backend_camera_id: null, label: '서쪽 카메라', rtsp_url_masked: 'rtsp://***', floor_name: '3층', space_id: 'shared', space_name: '공용실', status: 'online', created_at: null },
];

const clips: Clip[] = [
  { id: 'bed', camera_id: 'worker-a', camera_label: 'backend alias', event_type: 'bed-exit', created_at: '2026-07-20T01:00:00Z', label: null, reviewState: 'unknown', video_path: '/api/v1/clips/bed/video?token=secret', video_available: true, video_error: null },
  { id: 'fall', camera_id: 'local-b', camera_label: '서쪽 카메라', event_type: 'fall', created_at: null, label: null, reviewState: 'unknown', video_path: '/api/v1/clips/fall/video', video_available: false, video_error: '/var/clips/fall.mp4: ffmpeg failed' },
  { id: 'orphan', camera_id: 'removed', camera_label: '삭제된 카메라', event_type: 'fall', created_at: 'invalid', label: null, reviewState: 'unknown', video_path: '/api/v1/clips/orphan/video', video_available: false, video_error: 'secret backend detail' },
];

type RenderOptions = Partial<React.ComponentProps<typeof EventHistoryPage>>;

function renderPage(options: RenderOptions = {}) {
  const host = document.createElement('div');
  host.id = 'main-content';
  document.body.append(host);
  const root = createRoot(host);
  const props: React.ComponentProps<typeof EventHistoryPage> = {
    cameras,
    clips,
    status: 'success',
    lastSuccessAt: Date.parse('2026-07-20T02:00:00Z'),
    refreshing: false,
    location: { page: 'events' },
    onNavigate: vi.fn(),
    onRetry: vi.fn(),
    onClipChanged: vi.fn(),
    ...options,
  };
  act(() => root.render(<EventHistoryPage {...props} />));
  return { host, root, props };
}

afterEach(() => {
  document.body.innerHTML = '';
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('EventHistoryPage', () => {
  it('renders only a named loading state before the first successful list', () => {
    const { host, root } = renderPage({ clips: [], status: 'loading', lastSuccessAt: null });
    expect(host.textContent).toContain('이벤트 기록을 불러오는 중');
    expect(host.querySelectorAll('[data-testid="event-clip"]').length).toBe(0);
    act(() => root.unmount());
  });

  it('distinguishes successful empty data from loading', () => {
    const { host, root } = renderPage({ clips: [] });
    expect(host.textContent).toContain('저장된 이벤트 기록이 없습니다');
    expect(host.textContent).not.toContain('불러오는 중');
    act(() => root.unmount());
  });

  it('recovers from a first-load error into an empty successful result', () => {
    const rendered = renderPage({ clips: [], status: 'error', lastSuccessAt: null });
    expect(rendered.host.textContent).toContain('이벤트 기록을 불러오지 못했습니다');
    act(() => rendered.root.render(<EventHistoryPage {...rendered.props} clips={[]} status="success" />));
    expect(rendered.host.textContent).toContain('저장된 이벤트 기록이 없습니다');
    expect(rendered.host.textContent).not.toContain('불러오지 못했습니다');
    act(() => rendered.root.unmount());
  });

  it('retains stale clips after failure with last-good time and retry', () => {
    const onRetry = vi.fn();
    const { host, root } = renderPage({ status: 'error', onRetry });
    expect(host.textContent).toContain('최근 기록을 표시하고 있습니다');
    expect(host.textContent).toContain('마지막 확인');
    act(() => (Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '다시 시도') as HTMLButtonElement).click());
    expect(onRetry).toHaveBeenCalledOnce();
    act(() => root.unmount());
  });

  it('retains a successful empty result through refresh failure and clears the degraded state on recovery', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-21T04:05:06Z'));
    const lastSuccessAt = Date.now();
    const onRetry = vi.fn();
    const rendered = renderPage({ clips: [], lastSuccessAt, onRetry });

    act(() => rendered.root.render(
      <EventHistoryPage {...rendered.props} clips={[]} lastSuccessAt={lastSuccessAt} refreshing />,
    ));
    expect(rendered.host.textContent).toContain('이벤트 기록을 새로 확인하고 있습니다');
    expect(rendered.host.textContent).toContain('저장된 이벤트 기록이 없습니다');

    act(() => rendered.root.render(
      <EventHistoryPage {...rendered.props} clips={[]} status="error" lastSuccessAt={lastSuccessAt} refreshing={false} />,
    ));
    expect(rendered.host.querySelector('[data-testid="event-history-empty"]')?.textContent)
      .toContain('마지막 확인 당시 저장된 기록이 없습니다');
    expect(rendered.host.textContent).toContain(`마지막 확인 ${new Date(lastSuccessAt).toLocaleString('ko-KR')}`);
    expect(rendered.host.textContent).not.toContain('최근 기록을 표시하고 있습니다');
    act(() => Array.from(rendered.host.querySelectorAll('button')).find((button) => button.textContent === '다시 시도')?.click());
    expect(onRetry).toHaveBeenCalledOnce();

    const recoveredAt = Date.parse('2026-07-21T04:06:00Z');
    act(() => rendered.root.render(
      <EventHistoryPage
        {...rendered.props}
        clips={[clips[0]]}
        status="success"
        lastSuccessAt={recoveredAt}
        refreshing={false}
        location={{ page: 'events', clip: 'bed' }}
      />,
    ));
    expect(rendered.host.textContent).not.toContain('마지막 확인 당시 저장된 기록이 없습니다');
    expect(rendered.host.textContent).not.toContain('목록 갱신이 지연');
    expect(document.body.querySelector('video')?.getAttribute('src')).toBe('/api/v1/clips/bed/video?token=secret');
    expect(document.body.textContent).toContain('라벨 정보 없음');

    act(() => rendered.root.unmount());
  });

  it('round-trips exact filters and clears dependent URL fields through navigation updates', () => {
    const onNavigate = vi.fn();
    const location: EventHistoryLocation = { page: 'events', floor: '2층', room: 'shared', camera: 'local-a', event: 'bed-exit', clip: 'bed' };
    const { host, root } = renderPage({ location, onNavigate });
    const floor = host.querySelector('[aria-label="층 필터"]') as HTMLSelectElement;
    expect(floor.value).toBe('2층');
    act(() => { floor.value = '3층'; floor.dispatchEvent(new Event('change', { bubbles: true })); });
    expect(onNavigate).toHaveBeenCalledWith({ floor: '3층', room: null, camera: null, event: null, clip: null });
    act(() => root.unmount());
  });

  it('deep-links playable evidence with an accessible camera/event/time name and no secret copy', () => {
    const { host, root } = renderPage({ location: { page: 'events', clip: 'bed' } });
    const video = document.body.querySelector('video');
    expect(document.body.querySelector('[role="status"]')).toBeNull();
    expect(video?.getAttribute('src')).toContain('/api/v1/clips/bed/video');
    expect(video?.getAttribute('aria-label')).toContain('동쪽 카메라 침대 이탈 2026');
    expect(document.body.textContent).not.toContain('token=secret');
    act(() => root.unmount());
  });

  it('restores focus to the event heading after closing direct-linked evidence', () => {
    const onNavigate = vi.fn();
    const rendered = renderPage({ location: { page: 'events', clip: 'bed' }, onNavigate });
    const heading = rendered.host.querySelector('#event-history-heading');
    const closeButton = Array.from(document.body.querySelectorAll('button')).find((button) => button.textContent === '닫기');

    act(() => closeButton?.click());
    expect(onNavigate).toHaveBeenCalledWith({ clip: null });
    act(() => rendered.root.render(<EventHistoryPage {...rendered.props} location={{ page: 'events' }} />));

    expect(document.activeElement).toBe(heading);
    expect(document.activeElement).not.toBe(document.body);
    act(() => rendered.root.unmount());
  });

  it('restores focus to the event heading after escaping direct-linked evidence', () => {
    const onNavigate = vi.fn();
    const rendered = renderPage({ location: { page: 'events', clip: 'bed' }, onNavigate });
    const heading = rendered.host.querySelector('#event-history-heading');

    act(() => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
    expect(onNavigate).toHaveBeenCalledWith({ clip: null });
    act(() => rendered.root.render(<EventHistoryPage {...rendered.props} location={{ page: 'events' }} />));

    expect(document.activeElement).toBe(heading);
    expect(document.activeElement).not.toBe(document.body);
    act(() => rendered.root.unmount());
  });

  it('shows unavailable media without exposing backend paths or errors', () => {
    const { host, root } = renderPage({ location: { page: 'events', clip: 'fall' } });
    expect(document.body.querySelector('video')).toBeNull();
    expect(document.body.querySelector('[role="status"]')?.getAttribute('aria-label')).toContain('서쪽 카메라 낙상 시간 정보 없음');
    expect(document.body.textContent).toContain('영상을 불러올 수 없음');
    expect(document.body.textContent).not.toContain('/var/clips');
    expect(document.body.textContent).not.toContain('ffmpeg');
    act(() => root.unmount());
  });

  it('replaces a video that fails at runtime with a sanitized unavailable state', () => {
    const { root } = renderPage({ location: { page: 'events', clip: 'bed' } });
    const video = document.body.querySelector('video');

    act(() => video?.dispatchEvent(new Event('error')));

    expect(document.body.querySelector('video')).toBeNull();
    expect(document.body.querySelector('[role="status"]')?.textContent).toBe('영상을 불러올 수 없음');
    expect(document.body.textContent).not.toContain('token=secret');
    act(() => root.unmount());
  });

  it('recovers runtime playback after the source changes or another clip is selected', () => {
    const first = renderPage({ location: { page: 'events', clip: 'bed' } });
    act(() => document.body.querySelector('video')?.dispatchEvent(new Event('error')));
    expect(document.body.querySelector('video')).toBeNull();

    const refreshedBed = { ...clips[0], video_path: '/api/v1/clips/bed-v2/video' };
    act(() => first.root.render(<EventHistoryPage {...first.props} clips={[refreshedBed, clips[1], clips[2]]} />));
    expect(document.body.querySelector('video')?.getAttribute('src')).toBe('/api/v1/clips/bed-v2/video');

    act(() => document.body.querySelector('video')?.dispatchEvent(new Event('error')));
    const playableFall = { ...clips[1], video_available: true, video_error: null };
    act(() => first.root.render(
      <EventHistoryPage {...first.props} clips={[refreshedBed, playableFall, clips[2]]} location={{ page: 'events', clip: 'fall' }} />,
    ));
    expect(document.body.querySelector('video')?.getAttribute('src')).toBe('/api/v1/clips/fall/video');
    act(() => first.root.unmount());
  });

  it('describes orphan clips without inventing camera, location, or time metadata', () => {
    const { host, root } = renderPage({ location: { page: 'events' } });
    const orphan = Array.from(host.querySelectorAll('[data-testid="event-clip"]')).find((entry) => entry.textContent?.includes('카메라 정보 없음'));
    expect(orphan?.textContent).toContain('위치 정보 없음');
    expect(orphan?.textContent).toContain('시간 정보 없음');
    act(() => root.unmount());
  });

  it('displays an unsupported near-match event name without inventing a bed-exit label', () => {
    const warning = { ...clips[0], id: 'warning', event_type: 'bed-exit-warning' };
    const { host, root } = renderPage({ clips: [warning] });
    const card = host.querySelector('[data-testid="event-clip"]');
    expect(card?.querySelector('p')?.textContent).toBe('bed-exit-warning');
    expect(card?.textContent).not.toContain('침대 이탈');
    act(() => root.unmount());
  });

  it('renders unknown label truth and no live alert or heartbeat claims', () => {
    const { host, root } = renderPage({ location: { page: 'events', clip: 'bed' } });
    expect(host.querySelectorAll('[data-testid="event-location-group"]')).toHaveLength(3);
    expect(Array.from(host.querySelectorAll('h2')).map((heading) => heading.textContent)).toEqual(expect.arrayContaining(['2층 · 공용실', '3층 · 공용실']));
    expect(document.body.textContent).toContain('라벨 정보 없음');
    expect(document.body.textContent).not.toMatch(/Heartbeat|라이브 알림|실시간 상태|오버레이/);
    act(() => root.unmount());
  });

  it('preserves confirmed label metadata across list polling and resets to unknown on remount without it', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ clip_id: 'bed', label: null, reviewer: 'operator', reviewed_at: '2026-07-21T00:00:00Z' }) }));
    const { root, props } = renderPage({ location: { page: 'events', clip: 'bed' } });
    const unreviewed = Array.from(document.body.querySelectorAll('button')).find((button) => button.textContent === '미검토');
    await act(async () => { unreviewed?.click(); });
    expect(document.body.textContent).toContain('검토자 operator');
    act(() => root.render(<EventHistoryPage {...props} clips={[{ ...clips[0], video_path: '/api/v1/clips/bed-v2/video' }]} />));
    expect(document.body.querySelector('video')?.getAttribute('src')).toBe('/api/v1/clips/bed-v2/video');
    expect(document.body.textContent).toContain('검토자 operator');
    act(() => root.unmount());

    const reloaded = renderPage({ location: { page: 'events', clip: 'bed' }, clips: [{ ...clips[0], camera_label: 'reload metadata' }] });
    expect(document.body.textContent).toContain('라벨 정보 없음');
    expect(document.body.textContent).not.toContain('검토자 operator');
    act(() => reloaded.root.unmount());
  });

  it('keeps clip metadata when label saving fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500, json: async () => ({ detail: 'secret' }) }));
    const { root } = renderPage({ location: { page: 'events', clip: 'bed' } });
    const button = Array.from(document.body.querySelectorAll('button')).find((entry) => entry.textContent === '실제 침대 이탈');
    await act(async () => { button?.click(); });
    expect(document.body.textContent).toContain('라벨 저장에 실패했습니다');
    expect(document.body.textContent).toContain('동쪽 카메라');
    expect(document.body.querySelector('video')).not.toBeNull();
    act(() => root.unmount());
  });
});
