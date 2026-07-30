import { FormEvent, useEffect, useState } from 'react';
import { testCamera, updateCameraDecodeBackend, updateCamera, type Camera, type DecodeBackend } from '@/shared/api/client';
import { connectionFailureMessage } from '@/features/cameras/connectionTestMessage';
import { StatusBadge } from '@/shared/ui/StatusBadge';

const DECODE_BACKEND_OPTIONS: Array<{ value: DecodeBackend; label: string }> = [
  { value: 'auto', label: '자동 (GPU→CPU)' },
  { value: 'nvdec', label: 'GPU (NVDEC)' },
  { value: 'cpu', label: 'CPU' },
];

function toDecodeBackendValue(value: Camera['decode_backend']): DecodeBackend {
  if (value === 'nvdec' || value === 'cpu' || value === 'opencv') {
    return value === 'opencv' ? 'cpu' : value;
  }
  return 'auto';
}

type CameraCardProps = {
  camera: Camera;
  onUpdateStarted?: (camera: Camera) => number;
  onUpdated?: (camera: Camera, previousCameraId: string, startedAtGeneration?: number) => void;
  onDelete?: (camera: Camera) => void;
};

export function CameraCard({ camera, onUpdateStarted, onUpdated, onDelete }: CameraCardProps): JSX.Element {
  const mapped = Boolean(camera.space_id || camera.backend_camera_id);
  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState(camera.label);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [decodeBackend, setDecodeBackend] = useState<DecodeBackend>(toDecodeBackendValue(camera.decode_backend));
  const [decodeBusy, setDecodeBusy] = useState(false);
  const [testBusy, setTestBusy] = useState(false);

  function notifyUpdated(updated: Camera, startedAtGeneration: number | undefined): void {
    if (startedAtGeneration === undefined) {
      onUpdated?.(updated, camera.id);
    } else {
      onUpdated?.(updated, camera.id, startedAtGeneration);
    }
  }

  useEffect(() => {
    if (!editing && !busy) {
      setLabel(camera.label);
    }
  }, [busy, camera.label, editing]);

  useEffect(() => {
    if (!decodeBusy) {
      setDecodeBackend(toDecodeBackendValue(camera.decode_backend));
    }
  }, [camera.decode_backend, decodeBusy]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setMessage(null);
    if (!label.trim()) {
      setMessage('카메라 이름을 입력하세요.');
      return;
    }
    setBusy(true);
    const startedAtGeneration = onUpdateStarted?.(camera);
    try {
      const updated = await updateCamera(camera.id, {
        label: label.trim(),
      });
      notifyUpdated(updated, startedAtGeneration);
      setEditing(false);
      setMessage('카메라 정보를 수정했습니다.');
    } catch {
      setMessage('카메라 수정에 실패했습니다.');
    } finally {
      setBusy(false);
    }
  }

  async function handleDecodeBackendChange(value: DecodeBackend): Promise<void> {
    setDecodeBackend(value);
    setMessage(null);
    setDecodeBusy(true);
    const startedAtGeneration = onUpdateStarted?.(camera);
    try {
      const updated = await updateCameraDecodeBackend(camera.id, value);
      notifyUpdated(updated, startedAtGeneration);
      setMessage('디코딩 백엔드를 변경했습니다.');
    } catch {
      setDecodeBackend(toDecodeBackendValue(camera.decode_backend));
      setMessage('디코딩 백엔드 변경에 실패했습니다.');
    } finally {
      setDecodeBusy(false);
    }
  }

  function handleCancel(): void {
    if (busy) return;
    setLabel(camera.label);
    setEditing(false);
  }

  async function handleConnectionTest(): Promise<void> {
    if (testBusy) return;
    setMessage(null);
    setTestBusy(true);
    try {
      const result = await testCamera(camera.id);
      const size = result.width && result.height ? ` · ${result.width}×${result.height}` : '';
      setMessage(result.ok ? `연결 테스트 성공${size}` : `연결 테스트 실패 · ${connectionFailureMessage(result.error_class)}`);
    } catch {
      setMessage('연결 테스트 실패');
    } finally {
      setTestBusy(false);
    }
  }

  return (
    <article className="rounded-2xl border border-border bg-surface p-5 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-bold tracking-[0.24em] text-brand">카메라</p>
          <h3 className="mt-2 text-xl font-black text-ink">{camera.label}</h3>
        </div>
        <StatusBadge status={camera.status} />
      </div>

      <dl className="mt-5 space-y-3 text-sm">
        <div className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 rounded-2xl bg-surface2 px-3 py-2">
          <dt className="font-semibold text-ink-faint">위치</dt>
          <dd className="text-right font-bold text-ink-soft">{[camera.floor_name, camera.space_name ?? camera.space_id].filter(Boolean).join(' · ') || '미지정'}</dd>
        </div>
      </dl>
      <details className="mt-3 rounded-2xl bg-surface2 px-3 py-2 text-sm">
          <summary className="min-h-11 cursor-pointer py-3 font-semibold text-ink-soft">기술 설정</summary>
          <dl className="mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-2">
            <dt className="font-semibold text-ink-faint">등록 ID</dt>
            <dd className="break-all text-right font-mono text-xs text-ink-soft">{camera.id}</dd>
            <dt className="font-semibold text-ink-faint">관리 방식</dt>
            <dd className={mapped ? 'text-right font-bold text-brand' : 'text-right font-bold text-ink-faint'}>{mapped ? '서버 자동 관리' : '로컬 등록'}</dd>
          </dl>
          <div className="mt-3">
            <label className="sr-only" htmlFor={`decode-backend-${camera.id}`}>디코딩 백엔드</label>
            <select
              id={`decode-backend-${camera.id}`}
              value={decodeBackend}
              disabled={decodeBusy}
              onChange={(event) => void handleDecodeBackendChange(event.target.value as DecodeBackend)}
              className="min-h-11 w-full rounded-xl border border-border bg-surface px-3 py-2 text-sm font-bold text-ink outline-none ring-brand focus:ring-4 disabled:opacity-60"
            >
              {DECODE_BACKEND_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>
      </details>

      {editing ? (
        <form className="mt-5 space-y-3 rounded-xl bg-surface2 p-4" onSubmit={(event) => void handleSubmit(event)} noValidate>
          <label className="block text-sm font-bold text-ink-soft">
            이름
            <input name="label" value={label} onChange={(event) => setLabel(event.target.value)} className="mt-2 w-full rounded-2xl border border-border bg-surface px-4 py-3 text-ink outline-none ring-brand focus:ring-4" />
          </label>
          <div className="flex flex-wrap gap-2">
            <button type="submit" disabled={busy} className="brand-action rounded-lg px-4 py-2 text-xs font-black disabled:opacity-60">{busy ? '저장 중...' : '저장'}</button>
            <button type="button" disabled={busy} onClick={handleCancel} className="rounded-full bg-surface px-4 py-2 text-xs font-black text-ink-soft shadow-sm disabled:opacity-60">취소</button>
          </div>
        </form>
      ) : null}

      {message ? <p className="mt-4 text-xs font-bold text-brand" role="status">{message}</p> : null}

      <div className="mt-5 flex flex-wrap justify-between gap-2">
        <p className="text-xs text-ink-faint">등록 {camera.created_at ? new Date(camera.created_at).toLocaleString('ko-KR') : '정보 없음'}</p>
        <div className="flex gap-2">
          <button type="button" disabled={testBusy} onClick={() => void handleConnectionTest()} className="rounded-full bg-surface2 px-3 py-2 text-xs font-black text-ink-soft disabled:opacity-60">
            {testBusy ? '확인 중...' : '연결 테스트'}
          </button>
          <button type="button" disabled={busy} onClick={() => setEditing((value) => !value)} className="rounded-full bg-brand-soft px-3 py-2 text-xs font-black text-brand hover:bg-brand-soft disabled:opacity-60">
            수정
          </button>
          <button type="button" onClick={() => onDelete?.(camera)} className="rounded-lg bg-status-dangerBg px-3 py-2 text-xs font-black text-status-danger hover:bg-status-dangerBg">
            삭제
          </button>
        </div>
      </div>
    </article>
  );
}
