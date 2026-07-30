import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createCamera, testCamera } from '@/shared/api/client';
import { AddCameraModal } from '@/features/cameras/AddCameraModal';

vi.mock('@/shared/api/client', () => ({
  createCamera: vi.fn(),
  testCamera: vi.fn(),
}));

function setInput(host: HTMLElement, name: string, value: string): void {
  const input = host.querySelector(`input[name="${name}"]`);
  if (!(input instanceof HTMLInputElement)) {
    throw new Error(`missing input ${name}`);
  }
  act(() => {
    const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
    valueSetter?.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function renderModal(onCreated = vi.fn(), onClose = vi.fn()): { readonly host: HTMLDivElement; readonly root: ReturnType<typeof createRoot> } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => {
    root.render(<AddCameraModal open onClose={onClose} onCreated={onCreated} />);
  });
  return { host, root };
}

function fillValidForm(): void {
  setInput(document.body, 'label', '301호 침대 A');
  setInput(document.body, 'rtspHost', '192.0.2.10');
}

beforeEach(() => {
  vi.mocked(createCamera).mockReset();
  vi.mocked(testCamera).mockReset();
  vi.mocked(createCamera).mockResolvedValue({
    id: 'server-issued-camera-id',
    label: '301호 침대 A',
    rtsp_url_masked: 'rtsp://operator:***@192.0.2.10:8554/live',
    space_id: null,
    space_name: null,
    floor_name: null,
    backend_camera_id: null,
    status: 'offline',
    created_at: '2026-07-07T00:00:00.000Z',
  });
  vi.mocked(testCamera).mockResolvedValue({ ok: false, error_class: 'timeout' });
});

describe('AddCameraModal', () => {
  it('blocks submission and explains required camera label validation', () => {
    const { host, root } = renderModal();

    const submitButton = Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '카메라 등록');
    expect(submitButton?.classList.contains('brand-action')).toBe(true);

    act(() => {
      submitButton?.click();
    });

    expect(document.body.textContent).toContain('카메라 이름을 입력하세요.');

    act(() => root.unmount());
    host.remove();
  });

  it('creates a camera from name and structured RTSP fields without asking for a camera id or space_id', async () => {
    const { host, root } = renderModal();

    expect(document.body.textContent).not.toContain('space_id');
    expect(document.body.textContent).not.toContain('카메라 ID');
    expect((document.querySelector('input[name="rtspPath"]') as HTMLInputElement | null)?.value).toBe('/trackID=1');

    setInput(document.body, 'label', '301호 침대 A');
    setInput(document.body, 'rtspHost', '192.0.2.10');
    setInput(document.body, 'rtspPort', '8554');
    setInput(document.body, 'rtspUsername', 'operator');
    setInput(document.body, 'rtspPassword', 'secret');
    setInput(document.body, 'rtspQuery', 'profile=main');

    expect(document.body.textContent).toContain('rtsp://***@192.0.2.10:8554/trackID=1?profile=***');
    expect(document.body.textContent).not.toContain('operator');
    expect(document.body.textContent).not.toContain('secret');

    const submitButton = Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '카메라 등록');
    await act(async () => {
      submitButton?.click();
    });

    expect(createCamera).toHaveBeenCalledWith({
      label: '301호 침대 A',
      rtsp_url: 'rtsp://operator:secret@192.0.2.10:8554/trackID=1?profile=main',
    });
    expect(testCamera).toHaveBeenCalledWith('server-issued-camera-id');

    act(() => root.unmount());
    host.remove();
  });

  it.each([
    ['username only', 'operator', '', 'camera.local', 'rtsp://***@camera.local:554/trackID=1'],
    ['username and password', 'operator', 'secret', 'camera.local', 'rtsp://***@camera.local:554/trackID=1'],
    ['encoded credentials', 'operator name', 'p@ss/word', 'camera.local', 'rtsp://***@camera.local:554/trackID=1'],
    ['no credentials', '', '', 'camera.local', 'rtsp://camera.local:554/trackID=1'],
    ['malformed URL', 'operator', 'secret', '[invalid-host', 'rtsp://***@[invalid-host:554/trackID=1'],
  ])('masks the entire RTSP userinfo for %s', (_case, username, password, hostName, expectedPreview) => {
    const { host, root } = renderModal();
    setInput(document.body, 'rtspHost', hostName);
    setInput(document.body, 'rtspUsername', username);
    setInput(document.body, 'rtspPassword', password);

    const preview = document.querySelector('[data-testid="rtsp-preview"]');
    expect(preview?.textContent).toBe(expectedPreview);
    expect(preview?.closest('form')).not.toBeNull();
    expect(preview?.textContent).not.toContain('operator');
    expect(preview?.textContent).not.toContain('secret');
    expect(preview?.textContent).not.toContain('operator%20name');
    expect(preview?.textContent).not.toContain('p%40ss%2Fword');

    act(() => root.unmount());
    host.remove();
  });

  it.each([
    [
      'every ordinary and credential-shaped value',
      'profile=main&pwd=p%40ss&credential=scope&X-Goog-Signature=deadbeef&AWSAccessKeyId=AKIAEXAMPLE&camera=west',
      'profile=***&pwd=***&credential=***&X-Goog-Signature=***&AWSAccessKeyId=***&camera=***',
    ],
    [
      'repeated, encoded, empty, and key-only parameters',
      'profile=first&profile=second&%70rofile=encoded%2Fvalue&empty=&flag&bad%ZZ=value',
      'profile=***&profile=***&%70rofile=***&empty=***&***&bad%ZZ=***',
    ],
    [
      'values containing encoded and literal equals signs',
      'opaque=one%26two%3Dthree&signed=a=b=c',
      'opaque=***&signed=***',
    ],
  ])('redacts every query value for %s while preserving query key structure', (_case, query, expectedQuery) => {
    const { host, root } = renderModal();
    setInput(document.body, 'rtspHost', 'camera.local');
    setInput(document.body, 'rtspQuery', query);

    const preview = document.querySelector('[data-testid="rtsp-preview"]')?.textContent ?? '';
    expect(preview).toBe(`rtsp://camera.local:554/trackID=1?${expectedQuery}`);
    expect(['main', 'p%40ss', 'scope', 'deadbeef', 'AKIAEXAMPLE', 'encoded%2Fvalue', 'one%26two%3Dthree', 'a=b=c'].some((secret) => preview.includes(secret))).toBe(false);

    act(() => root.unmount());
    host.remove();
  });

  it.each([
    [
      'mixed values, multiple key-only segments, and a fragment',
      'camera.local',
      'operator',
      'camera-password',
      'profile=main&opaque-secret-token&another%2Dsecret#token=fragment-secret',
      'rtsp://***@camera.local:554/trackID=1?profile=***&***&***#***',
      'rtsp://operator:camera-password@camera.local:554/trackID=1?profile=main&opaque-secret-token&another%2Dsecret#token=fragment-secret',
    ],
    [
      'encoded fragment on a malformed URL',
      '[invalid-host',
      '',
      '',
      'first-secret&second%20secret#encoded%2Ffragment%3Dvalue',
      'rtsp://[invalid-host:554/trackID=1?***&***#***',
      'rtsp://[invalid-host:554/trackID=1?first-secret&second%20secret#encoded%2Ffragment%3Dvalue',
    ],
    [
      'a single key-only query segment',
      'camera.local',
      '',
      '',
      'opaque-secret-token',
      'rtsp://camera.local:554/trackID=1?***',
      'rtsp://camera.local:554/trackID=1?opaque-secret-token',
    ],
  ])('redacts %s without changing the submitted RTSP URL', async (_case, hostName, username, password, query, expectedPreview, expectedSubmittedUrl) => {
    const { host, root } = renderModal();
    setInput(document.body, 'label', 'Privacy camera');
    setInput(document.body, 'rtspHost', hostName);
    setInput(document.body, 'rtspUsername', username);
    setInput(document.body, 'rtspPassword', password);
    setInput(document.body, 'rtspQuery', query);

    const preview = document.querySelector('[data-testid="rtsp-preview"]')?.textContent ?? '';
    expect(preview).toBe(expectedPreview);
    expect(['opaque-secret-token', 'another%2Dsecret', 'fragment-secret', 'first-secret', 'second%20secret', 'encoded%2Ffragment%3Dvalue'].some((secret) => preview.includes(secret))).toBe(false);

    await act(async () => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '카메라 등록')?.click());
    expect(vi.mocked(createCamera).mock.calls[0]?.[0].rtsp_url).toBe(expectedSubmittedUrl);

    act(() => root.unmount());
    host.remove();
  });

  it('keeps userinfo and query secrets masked for a malformed URL without changing the submitted RTSP value', async () => {
    const { host, root } = renderModal();
    setInput(document.body, 'label', 'Malformed camera');
    setInput(document.body, 'rtspHost', '[invalid-host');
    setInput(document.body, 'rtspUsername', 'operator');
    setInput(document.body, 'rtspPassword', 'camera-password');
    setInput(document.body, 'rtspQuery', 'profile=main&token=query-secret&token=second-secret');

    const preview = document.querySelector('[data-testid="rtsp-preview"]')?.textContent ?? '';
    expect(preview).toBe('rtsp://***@[invalid-host:554/trackID=1?profile=***&token=***&token=***');
    expect(['operator', 'camera-password', 'main', 'query-secret', 'second-secret'].some((secret) => preview.includes(secret))).toBe(false);

    await act(async () => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '카메라 등록')?.click());
    const submittedUrl = vi.mocked(createCamera).mock.calls[0]?.[0].rtsp_url;
    expect(submittedUrl === 'rtsp://operator:camera-password@[invalid-host:554/trackID=1?profile=main&token=query-secret&token=second-secret').toBe(true);

    act(() => root.unmount());
    host.remove();
  });

  it('completes registration after a successful follow-up probe', async () => {
    const onClose = vi.fn();
    const onCreated = vi.fn();
    vi.mocked(testCamera).mockResolvedValue({ ok: true, width: 1280, height: 720 });
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<AddCameraModal open onClose={onClose} onCreated={onCreated} />));
    fillValidForm();

    await act(async () => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '카메라 등록')?.click());
    expect(onCreated).toHaveBeenCalledWith(expect.objectContaining({ id: 'server-issued-camera-id' }));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).not.toContain('등록 완료 · 연결 확인 실패');
    act(() => root.unmount());
    host.remove();
  });

  it('keeps values on validation and create failure, validates the port, and suppresses repeated submit', async () => {
    let rejectCreate: ((reason?: unknown) => void) | undefined;
    vi.mocked(createCamera).mockImplementation(() => new Promise((_, reject) => { rejectCreate = reject; }));
    const onCreated = vi.fn();
    const { host, root } = renderModal(onCreated);
    fillValidForm();
    setInput(document.body, 'rtspPort', '70000');

    const submit = Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '카메라 등록');
    act(() => submit?.click());
    expect(document.body.textContent).toContain('1부터 65535');
    expect((document.querySelector('input[name="label"]') as HTMLInputElement).value).toBe('301호 침대 A');

    setInput(document.body, 'rtspPort', '554');
    act(() => { submit?.click(); submit?.click(); });
    expect(createCamera).toHaveBeenCalledTimes(1);
    await act(async () => rejectCreate?.(new Error('Invalid camera response')));
    expect(document.body.textContent).toContain('카메라 등록에 실패했습니다.');
    expect(document.body.textContent).toContain('다시 시도');
    expect(document.body.textContent).not.toContain('API 상태');
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();
    expect((document.querySelector('input[name="label"]') as HTMLInputElement).value).toBe('301호 침대 A');
    expect(onCreated).not.toHaveBeenCalled();
    expect(testCamera).not.toHaveBeenCalled();

    act(() => root.unmount());
    host.remove();
  });

  it.each(['1e3', '0x50', ' 554', '554 ', '+554', '-554', '554.0', '５５４'])('rejects non-ASCII-decimal RTSP port input %j', (port) => {
    const { host, root } = renderModal();
    fillValidForm();
    setInput(document.body, 'rtspPort', port);

    act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '카메라 등록')?.click());

    expect(document.body.textContent).toContain('1부터 65535');
    expect(createCamera).not.toHaveBeenCalled();

    act(() => root.unmount());
    host.remove();
  });

  it('keeps a pending create dialog open across every close attempt and prevents a duplicate submit', async () => {
    let resolveCreate: ((camera: Awaited<ReturnType<typeof createCamera>>) => void) | undefined;
    vi.mocked(createCamera).mockImplementation(() => new Promise((resolve) => { resolveCreate = resolve; }));
    vi.mocked(testCamera).mockResolvedValue({ ok: true, width: 640, height: 480 });
    const onClose = vi.fn();
    const onCreated = vi.fn();
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<AddCameraModal open onClose={onClose} onCreated={onCreated} />));
    fillValidForm();

    act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '카메라 등록')?.click());
    const close = Array.from(document.querySelectorAll<HTMLButtonElement>('[role="dialog"] button')).find((button) => button.textContent === '닫기');
    expect(close?.disabled).toBe(true);
    act(() => close?.click());
    act(() => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
    act(() => document.querySelector('[data-dialog-backdrop]')?.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '등록 중...')?.click());

    expect(onClose).not.toHaveBeenCalled();
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();
    expect(createCamera).toHaveBeenCalledTimes(1);

    await act(async () => resolveCreate?.({
      id: 'server-issued-camera-id',
      label: '301호 침대 A',
      rtsp_url_masked: 'rtsp://operator:***@192.0.2.10:8554/live',
      space_id: null,
      space_name: null,
      floor_name: null,
      backend_camera_id: null,
      status: 'offline',
      created_at: '2026-07-07T00:00:00.000Z',
    }));
    expect(onCreated).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
    host.remove();
  });

  it('preserves a successful create when probing fails and retries only the probe', async () => {
    const onCreated = vi.fn();
    vi.mocked(testCamera).mockRejectedValueOnce(new Error('timeout')).mockResolvedValueOnce({ ok: true, width: 1280, height: 720 });
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<AddCameraModal open onClose={vi.fn()} onCreated={onCreated} />));
    fillValidForm();

    await act(async () => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '카메라 등록')?.click());
    expect(onCreated).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).toContain('등록 완료 · 연결 확인 실패');
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();
    await act(async () => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '연결 확인 재시도')?.click());
    expect(createCamera).toHaveBeenCalledTimes(1);
    expect(testCamera).toHaveBeenCalledTimes(2);

    act(() => root.unmount());
    host.remove();
  });

  it.each([
    ['timeout', '카메라 응답 시간이 초과되었습니다. 네트워크와 전원을 확인하세요.'],
    ['auth', '카메라 인증에 실패했습니다. 아이디와 비밀번호를 확인하세요.'],
    ['decode', '영상 형식을 처리할 수 없습니다. 카메라 스트림 설정을 확인하세요.'],
    [undefined, '카메라 연결을 확인할 수 없습니다. 네트워크와 설정을 확인하세요.'],
  ] as const)('given %s probe failure, when registration succeeds, then shows a sanitized operator message', async (errorClass, expected) => {
    vi.mocked(testCamera).mockResolvedValue({ ok: false, error_class: errorClass });
    const { host, root } = renderModal();
    fillValidForm();

    await act(async () => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '카메라 등록')?.click());

    expect(document.body.textContent).toContain(expected);
    if (errorClass) expect(document.body.textContent).not.toContain(errorClass);
    act(() => root.unmount());
    host.remove();
  });

  it('uses the bounded shared dialog surface at a 375px viewport', () => {
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 375 });
    const { host, root } = renderModal();
    expect(document.querySelector('.accessible-dialog')).not.toBeNull();
    expect(document.querySelector('[data-dialog-backdrop]')).not.toBeNull();
    expect(document.body.scrollWidth).toBeLessThanOrEqual(375);
    act(() => root.unmount());
    host.remove();
  });

  it('initially focuses the camera name input instead of the close button', () => {
    const { host, root } = renderModal();
    expect(document.activeElement).toBe(document.querySelector('input[name="label"]'));
    act(() => root.unmount());
    host.remove();
  });
});
