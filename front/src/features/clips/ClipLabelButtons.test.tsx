import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Clip } from '@/shared/api/client';
import { ClipLabelButtons, koreanClipLabel } from '@/features/clips/ClipLabelButtons';

const clip: Clip = {
  id: 'clip-1',
  camera_id: 'cam-1',
  camera_label: '301호',
  event_type: 'fall',
  created_at: '2026-07-06T00:00:00.000Z',
  label: null,
  reviewer: null,
  reviewed_at: null,
  reviewState: 'unknown',
  video_path: '/api/v1/clips/clip-1/video',
  video_available: true,
  video_error: null,
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ClipLabelButtons', () => {
  it.each([
    ['fall', '실제 낙상'],
    [' FALL ', '실제 낙상'],
    ['bed-exit', '실제 침대 이탈'],
    ['BED_EXIT', '실제 침대 이탈'],
    ['bed exit', '실제 침대 이탈'],
  ])('uses the explicit supported event type %j for TRUE_POSITIVE labels', (eventType, expected) => {
    expect(koreanClipLabel('TRUE_POSITIVE', 'confirmed', eventType)).toBe(expected);
  });

  it.each([
    'not_fall',
    'fall-risk',
    'night-bed-exit',
    'bed-exit-warning',
    '',
    undefined,
  ])('uses a truthful generic TRUE_POSITIVE label for unsupported event type %j', (eventType) => {
    expect(koreanClipLabel('TRUE_POSITIVE', 'confirmed', eventType)).toBe('실제 이벤트');
  });

  it('renders an unknown server label honestly instead of assuming unreviewed', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<ClipLabelButtons clip={clip} onChanged={() => undefined} />));
    expect(host.textContent).toContain('라벨 정보 없음');
    expect(Array.from(host.querySelectorAll('button')).every((button) => button.getAttribute('aria-pressed') === 'false')).toBe(true);
    act(() => root.unmount());
    host.remove();
  });

  it('selects 미검토 for a confirmed null label and exposes one pressed choice', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(
      <ClipLabelButtons clip={{ ...clip, label: null, reviewState: 'confirmed' }} onChanged={() => undefined} />,
    ));

    const buttons = Array.from(host.querySelectorAll('button'));
    expect(buttons.filter((button) => button.getAttribute('aria-pressed') === 'true')).toHaveLength(1);
    expect(buttons.find((button) => button.textContent === '미검토')?.getAttribute('aria-pressed')).toBe('true');

    act(() => root.unmount());
    host.remove();
  });

  it('renders bed-exit confirmation without calling it a fall', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(
      <ClipLabelButtons clip={{ ...clip, event_type: 'bed-exit' }} onChanged={() => undefined} />,
    ));

    expect(Array.from(host.querySelectorAll('button')).map((button) => button.textContent)).toContain('실제 침대 이탈');
    expect(host.textContent).not.toContain('실제 낙상');

    act(() => root.unmount());
    host.remove();
  });

  it('sends TRUE_POSITIVE label to the clip label endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ clip_id: clip.id, label: 'TRUE_POSITIVE', reviewer: 'operator', reviewed_at: '2026-07-07T00:00:00Z' }),
    });
    vi.stubGlobal('fetch', fetchMock);
    const onChanged = vi.fn();
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);

    act(() => {
      root.render(<ClipLabelButtons clip={clip} onChanged={onChanged} />);
    });

    const button = Array.from(host.querySelectorAll('button')).find((entry) => entry.textContent === '실제 낙상');
    await act(async () => {
      button?.click();
    });

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/clips/clip-1/label', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ label: 'TRUE_POSITIVE' }),
    }));
    expect(onChanged).toHaveBeenCalledWith(expect.objectContaining({ label: 'TRUE_POSITIVE', camera_id: 'cam-1', reviewState: 'confirmed' }));
    expect(host.textContent).toContain('실제 낙상 라벨을 저장했습니다.');

    act(() => root.unmount());
    host.remove();
  });
});
