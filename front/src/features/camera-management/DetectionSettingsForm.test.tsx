import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { describe, expect, it, vi } from 'vitest';
import type { Camera } from '@/shared/api/client';
import { buildDetectionSettingsPatch, DetectionSettingsForm } from '@/features/camera-management/DetectionSettingsForm';

const camera: Camera = {
  id: 'cam-1',
  label: '301호 침대 A',
  rtsp_url_masked: 'rtsp://camera/stream',
  space_id: 'space-301',
  space_name: null,
  floor_name: null,
  backend_camera_id: null,
  status: 'online',
  created_at: '2026-07-06T00:00:00.000Z',
};

describe('DetectionSettingsForm', () => {
  it('builds PATCH /cameras extension payload shape', () => {
    expect(buildDetectionSettingsPatch('cam-1', {
      threshold: 0.8,
      fallEnabled: true,
      bedExitEnabled: false,
      bedCount: 2,
      nightStart: '21:00',
      nightEnd: '05:30',
    })).toEqual({
      cameraId: 'cam-1',
      payload: {
        detectionSettings: {
          threshold: 0.8,
          domains: { fall: true, bed_exit: false },
          bedCount: 2,
          nightWindow: { start: '21:00', end: '05:30' },
        },
      },
    });
  });

  it('shows supported-later notice instead of a fake save', () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);

    act(() => {
      root.render(<DetectionSettingsForm cameras={[camera]} />);
    });

    const button = Array.from(host.querySelectorAll('button')).find((entry) => entry.textContent === '설정 저장');
    act(() => {
      button?.click();
    });

    expect(host.textContent).toContain('저장 지원 예정');
    expect(host.textContent).toContain('가짜 성공으로 처리하지 않았습니다');
    expect(fetchMock).not.toHaveBeenCalled();

    act(() => root.unmount());
    host.remove();
    vi.unstubAllGlobals();
  });
});
