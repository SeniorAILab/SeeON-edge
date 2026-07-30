import { FormEvent, useMemo, useRef, useState } from 'react';
import { createCamera, testCamera, type Camera } from '@/shared/api/client';
import { connectionFailureMessage } from '@/features/cameras/connectionTestMessage';
import {
  buildCameraRegistrationRtspUrl,
  initialCameraRegistrationRtspForm,
  maskCameraRegistrationRtspPreview,
  validateCameraRegistrationForm,
  type CameraRegistrationRtspForm,
} from '@/features/cameras/cameraRegistrationForm';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';

type AddCameraModalProps = {
  open: boolean;
  onClose: () => void;
  onCreated: (camera: Camera) => void;
};

export function AddCameraModal({ open, onClose, onCreated }: AddCameraModalProps): JSX.Element | null {
  const [label, setLabel] = useState('');
  const [rtspForm, setRtspForm] = useState<CameraRegistrationRtspForm>(initialCameraRegistrationRtspForm);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const nameInputRef = useRef<HTMLInputElement>(null);
  const [createdCamera, setCreatedCamera] = useState<Camera | null>(null);

  const rtspUrl = useMemo(() => buildCameraRegistrationRtspUrl(rtspForm), [rtspForm]);
  const rtspPreview = useMemo(() => maskCameraRegistrationRtspPreview(rtspUrl), [rtspUrl]);
  const validationError = useMemo(() => validateCameraRegistrationForm(label, rtspForm), [label, rtspForm]);

  function resetAndClose(): void {
    setLabel('');
    setRtspForm(initialCameraRegistrationRtspForm);
    setMessage(null);
    setBusy(false);
    busyRef.current = false;
    setCreatedCamera(null);
    onClose();
  }

  function requestClose(): void {
    if (busyRef.current) return;
    resetAndClose();
  }

  function updateRtspField(field: keyof CameraRegistrationRtspForm, value: string): void {
    setRtspForm((current) => ({ ...current, [field]: value }));
  }

  async function handleCreate(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (busyRef.current || createdCamera) return;
    setMessage(null);
    if (validationError) {
      setMessage(validationError);
      return;
    }

    busyRef.current = true;
    setBusy(true);
    try {
      const camera = await createCamera({ label: label.trim(), rtsp_url: rtspUrl });
      onCreated(camera);
      setCreatedCamera(camera);
      await probeCreatedCamera(camera);
    } catch {
      setMessage('카메라 등록에 실패했습니다. 잠시 후 다시 시도하고 서비스 연결 상태를 확인하세요.');
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  async function probeCreatedCamera(camera: Camera): Promise<void> {
    try {
      const result = await testCamera(camera.id);
      if (!result.ok) {
        setMessage(`등록 완료 · 연결 확인 실패 · ${connectionFailureMessage(result.error_class)}`);
        return;
      }
      resetAndClose();
    } catch {
      setMessage('등록 완료 · 연결 확인 실패');
    }
  }

  async function retryProbe(): Promise<void> {
    if (!createdCamera || busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setMessage(null);
    try {
      await probeCreatedCamera(createdCamera);
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  return (
    <AccessibleDialog open={open} title="카메라 추가" onClose={requestClose} initialFocusRef={nameInputRef}>
      <div className="flex items-start justify-between gap-4">
        <p className="text-sm font-bold text-brand">새 스트림</p>
        <button type="button" disabled={busy} onClick={requestClose} className="rounded-full bg-surface2 px-4 py-2 text-sm font-bold text-ink-soft hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-60">
          닫기
        </button>
      </div>

      <form className="mt-6 space-y-4" onSubmit={(event) => void handleCreate(event)} noValidate>
          <label className="block text-sm font-bold text-ink-soft">
            카메라 이름
            <input
              ref={nameInputRef}
              name="label"
              value={label}
              onChange={(event) => setLabel(event.target.value)}
              className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
              placeholder="예: 301호 침대 A"
            />
          </label>

          <div className="grid gap-3 sm:grid-cols-[120px_minmax(0,1fr)_120px]">
            <label className="block text-sm font-bold text-ink-soft">
              방식
              <select
                name="rtspScheme"
                value={rtspForm.scheme}
                onChange={(event) => updateRtspField('scheme', event.target.value === 'rtsps' ? 'rtsps' : 'rtsp')}
                className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
              >
                <option value="rtsp">rtsp</option>
                <option value="rtsps">rtsps</option>
              </select>
            </label>
            <label className="block text-sm font-bold text-ink-soft">
              카메라 IP/호스트
              <input
                name="rtspHost"
                value={rtspForm.host}
                onChange={(event) => updateRtspField('host', event.target.value)}
                className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
                placeholder="192.0.2.10"
              />
            </label>
            <label className="block text-sm font-bold text-ink-soft">
              포트
              <input
                name="rtspPort"
                inputMode="numeric"
                value={rtspForm.port}
                onChange={(event) => updateRtspField('port', event.target.value)}
                className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
              />
            </label>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <label className="block text-sm font-bold text-ink-soft">
              RTSP 아이디
              <input name="rtspUsername" value={rtspForm.username} onChange={(event) => updateRtspField('username', event.target.value)} className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4" />
            </label>
            <label className="block text-sm font-bold text-ink-soft">
              RTSP 비밀번호
              <input name="rtspPassword" type="password" value={rtspForm.password} onChange={(event) => updateRtspField('password', event.target.value)} className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4" />
            </label>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <label className="block text-sm font-bold text-ink-soft">
              경로
              <input name="rtspPath" value={rtspForm.path} onChange={(event) => updateRtspField('path', event.target.value)} className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4" placeholder="/trackID=1" />
            </label>
            <label className="block text-sm font-bold text-ink-soft">
              추가 인자
              <input name="rtspQuery" value={rtspForm.query} onChange={(event) => updateRtspField('query', event.target.value)} className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4" placeholder="profile=main" />
            </label>
          </div>

          <div className="rounded-2xl bg-surface2 px-4 py-3">
            <p className="text-xs font-black uppercase tracking-[0.18em] text-ink-faint">RTSP 미리보기</p>
            <p data-testid="rtsp-preview" className="mt-2 break-all font-mono text-xs text-ink-soft">{rtspPreview}</p>
          </div>

          <button type="submit" disabled={busy || Boolean(createdCamera)} className="brand-action rounded-lg px-6 py-3 text-sm font-black shadow-sm disabled:cursor-not-allowed disabled:opacity-60">
            {busy ? '등록 중...' : '카메라 등록'}
          </button>
      </form>

      {message ? (
        <p className="mt-5 rounded-2xl bg-brand-soft px-4 py-3 text-sm font-bold text-brand" role="status">
          {message}
        </p>
      ) : null}
      {createdCamera ? (
        <button type="button" disabled={busy} onClick={() => void retryProbe()} className="brand-action mt-3 rounded-full px-5 py-3 text-sm font-black disabled:opacity-60">
          연결 확인 재시도
        </button>
      ) : null}
    </AccessibleDialog>
  );
}
