import { FormEvent, useMemo, useState } from 'react';
import type { Camera, CameraPatchInput } from '@/shared/api/client';

type DetectionSettingsFormProps = {
  cameras: Camera[];
};

export function buildDetectionSettingsPatch(cameraId: string, values: {
  threshold: number;
  fallEnabled: boolean;
  bedExitEnabled: boolean;
  bedCount: number;
  nightStart: string;
  nightEnd: string;
}): { cameraId: string; payload: CameraPatchInput } {
  return {
    cameraId,
    payload: {
      detectionSettings: {
        threshold: values.threshold,
        domains: {
          fall: values.fallEnabled,
          bed_exit: values.bedExitEnabled,
        },
        bedCount: values.bedCount,
        nightWindow: {
          start: values.nightStart,
          end: values.nightEnd,
        },
      },
    },
  };
}

export function DetectionSettingsForm({ cameras }: DetectionSettingsFormProps): JSX.Element {
  const [cameraId, setCameraId] = useState(cameras[0]?.id ?? '');
  const [threshold, setThreshold] = useState('0.72');
  const [fallEnabled, setFallEnabled] = useState(true);
  const [bedExitEnabled, setBedExitEnabled] = useState(true);
  const [bedCount, setBedCount] = useState('1');
  const [nightStart, setNightStart] = useState('22:00');
  const [nightEnd, setNightEnd] = useState('06:00');
  const [message, setMessage] = useState<string | null>(null);

  const selectedCameraId = cameraId || cameras[0]?.id || '';
  const patchPreview = useMemo(() => {
    if (!selectedCameraId) {
      return null;
    }
    return buildDetectionSettingsPatch(selectedCameraId, {
      threshold: Number(threshold),
      fallEnabled,
      bedExitEnabled,
      bedCount: Number(bedCount),
      nightStart,
      nightEnd,
    });
  }, [bedCount, bedExitEnabled, fallEnabled, nightEnd, nightStart, selectedCameraId, threshold]);

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    if (!selectedCameraId) {
      setMessage('설정을 적용할 카메라가 없습니다. 카메라를 먼저 등록하세요.');
      return;
    }
    if (!Number.isFinite(Number(threshold)) || Number(threshold) < 0 || Number(threshold) > 1) {
      setMessage('threshold는 0부터 1 사이 숫자여야 합니다.');
      return;
    }
    if (!Number.isInteger(Number(bedCount)) || Number(bedCount) < 0) {
      setMessage('침대 갯수는 0 이상의 정수여야 합니다.');
      return;
    }
    setMessage('저장 지원 예정: 백엔드가 PATCH /api/v1/cameras/{id} detectionSettings 확장 필드를 수용하면 전송됩니다. 가짜 성공으로 처리하지 않았습니다.');
  }

  return (
    <form className="rounded-4xl bg-surface/80 p-6 shadow-soft" onSubmit={handleSubmit} noValidate>
      <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-sm font-black uppercase tracking-[0.2em] text-brand">Detection policy</p>
          <h2 className="mt-1 text-2xl font-black text-ink">탐지 설정</h2>
          <p className="mt-2 text-sm text-ink-soft">카메라별 threshold, 도메인, 침대 수, 야간 시간창을 설계합니다.</p>
        </div>
        <button type="submit" className="rounded-full bg-slate-950 px-5 py-3 text-sm font-black text-white hover:bg-brand">
          설정 저장
        </button>
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <label className="block text-sm font-bold text-ink-soft">
          카메라
          <select
            value={selectedCameraId}
            onChange={(event) => setCameraId(event.target.value)}
            className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
          >
            {cameras.length > 0 ? cameras.map((camera) => <option key={camera.id} value={camera.id}>{camera.label}</option>) : <option value="">카메라 없음</option>}
          </select>
        </label>
        <label className="block text-sm font-bold text-ink-soft">
          Threshold
          <input
            type="number"
            min="0"
            max="1"
            step="0.01"
            value={threshold}
            onChange={(event) => setThreshold(event.target.value)}
            className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
          />
        </label>
        <label className="block text-sm font-bold text-ink-soft">
          침대 갯수
          <input
            type="number"
            min="0"
            step="1"
            value={bedCount}
            onChange={(event) => setBedCount(event.target.value)}
            className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
          />
        </label>
        <div className="grid grid-cols-2 gap-3">
          <label className="block text-sm font-bold text-ink-soft">
            야간 시작
            <input type="time" value={nightStart} onChange={(event) => setNightStart(event.target.value)} className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4" />
          </label>
          <label className="block text-sm font-bold text-ink-soft">
            야간 종료
            <input type="time" value={nightEnd} onChange={(event) => setNightEnd(event.target.value)} className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4" />
          </label>
        </div>
      </div>

      <fieldset className="mt-5 rounded-3xl bg-surface2 p-4">
        <legend className="px-2 text-sm font-black text-ink-soft">도메인 on/off</legend>
        <div className="mt-2 flex flex-wrap gap-4 text-sm font-bold text-ink-soft">
          <label className="flex items-center gap-2"><input type="checkbox" checked={fallEnabled} onChange={(event) => setFallEnabled(event.target.checked)} /> 낙상 감지</label>
          <label className="flex items-center gap-2"><input type="checkbox" checked={bedExitEnabled} onChange={(event) => setBedExitEnabled(event.target.checked)} /> 침대 이탈</label>
        </div>
      </fieldset>

      {message ? <p className="mt-5 rounded-2xl bg-amber-50 px-4 py-3 text-sm font-black text-amber-800" role="status">{message}</p> : null}
      {patchPreview ? (
        <pre className="mt-5 overflow-auto rounded-3xl bg-slate-950 p-4 text-xs text-slate-100">
          {JSON.stringify(patchPreview, null, 2)}
        </pre>
      ) : null}
    </form>
  );
}
