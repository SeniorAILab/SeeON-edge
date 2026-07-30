import type { Camera } from '@/shared/api/client';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';

export function DeleteCameraDialog({
  camera,
  message,
  onCancel,
  onConfirm,
  busy = false,
}: {
  camera: Camera;
  message: string | null;
  onCancel: () => void;
  onConfirm: () => void;
  busy?: boolean;
}): JSX.Element {
  return (
    <AccessibleDialog open title={`${camera.label} 삭제`} onClose={onCancel}>
      <p className="text-sm font-black text-status-danger">삭제 확인</p>
      <p className="mt-3 text-sm leading-6 text-ink-soft">카메라를 삭제하면 연결된 상태 갱신이 중단됩니다. 서버가 참조 중인 이벤트를 보호하면 삭제가 거절될 수 있습니다.</p>
      {message ? <p className="mt-4 rounded-2xl bg-status-dangerBg px-4 py-3 text-sm font-bold text-status-danger" role="status">{message}</p> : null}
      <div className="mt-6 flex justify-end gap-3">
        <button type="button" onClick={onCancel} className="rounded-full bg-surface2 px-5 py-3 text-sm font-black text-ink-soft">취소</button>
        <button type="button" disabled={busy} onClick={onConfirm} className="rounded-full bg-status-danger px-5 py-3 text-sm font-black text-surface disabled:opacity-60">{busy ? '삭제 중...' : '삭제'}</button>
      </div>
    </AccessibleDialog>
  );
}
