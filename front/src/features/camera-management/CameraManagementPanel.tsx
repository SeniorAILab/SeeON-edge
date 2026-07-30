import type { Camera, CameraRegistry } from '@/shared/api/client';
import { CameraCard } from '@/features/cameras/CameraCard';

export function CameraManagementPanel({
  registry,
  cameraError,
  lastSuccessAt,
  refreshing,
  onAddCamera,
  onUpdateStarted,
  onUpdated,
  onDelete,
  loading = false,
  hasLoadedSuccessfully = true,
  onRetry,
}: {
  registry: CameraRegistry;
  cameraError: string | null;
  lastSuccessAt: number | null;
  refreshing: boolean;
  onAddCamera: () => void;
  onUpdateStarted: (camera: Camera) => number;
  onUpdated: (camera: Camera, previousCameraId?: string, startedAtGeneration?: number) => void;
  onDelete: (camera: Camera) => void;
  loading?: boolean;
  hasLoadedSuccessfully?: boolean;
  onRetry?: () => void;
}): JSX.Element {
  return (
    <section className="rounded-2xl border border-border bg-surface p-5 shadow-sm">
      <div className="mb-5 flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-black text-brand">카메라 연결 설정</p>
          <h2 className="text-2xl font-black text-ink">카메라 관리</h2>
        </div>
        <button
          type="button"
          onClick={onAddCamera}
          className="brand-action rounded-lg px-5 py-3 text-sm font-black"
        >
          + 카메라 추가
        </button>
      </div>
      <details className="mb-4 text-xs text-ink-faint">
        <summary className="min-h-11 cursor-pointer py-3 font-bold">기술 정보</summary>
        <p className="mt-2">등록 버전 {registry.registry_version}</p>
      </details>
      {cameraError ? (
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-2xl bg-status-dangerBg px-4 py-3 text-sm font-bold text-status-danger" role="alert">
          <div>
            <p>{cameraError}</p>
            {lastSuccessAt !== null ? <p className="mt-1 text-xs">마지막 확인 {new Date(lastSuccessAt).toLocaleString('ko-KR')}</p> : null}
          </div>
          {onRetry ? <button type="button" onClick={onRetry} className="rounded-full bg-surface px-4 py-2 text-xs font-black">다시 시도</button> : null}
        </div>
      ) : null}
      {refreshing && !loading && cameraError === null ? (
        <p className="mb-4 text-sm font-bold text-ink-soft" role="status" aria-live="polite">카메라 목록을 새로 확인하고 있습니다.</p>
      ) : null}
      {loading ? (
        <p className="rounded-2xl bg-surface px-5 py-8 text-center font-bold text-ink-soft" role="status">카메라 목록을 불러오는 중...</p>
      ) : registry.cameras.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2">
          {registry.cameras.map((camera) => (
            <CameraCard key={camera.id} camera={camera} onUpdateStarted={onUpdateStarted} onUpdated={onUpdated} onDelete={onDelete} />
          ))}
        </div>
      ) : hasLoadedSuccessfully ? (
        <div className="rounded-xl border border-dashed border-brand bg-surface p-8 text-center text-ink-soft">
          등록된 카메라가 없습니다. 첫 카메라를 추가하세요.
        </div>
      ) : null}
    </section>
  );
}
