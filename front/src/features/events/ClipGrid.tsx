import { ClipCard } from '@/features/events/ClipCard';
import type { Clip } from '@/shared/api/types';
import type { PollingResourceStatus } from '@/shared/api/usePollingResource';

type ClipGridProps = {
  status: PollingResourceStatus;
  hasData: boolean;
  clips: readonly Clip[];
  resolveCameraLabel: (clip: Clip) => string;
  onRetry: () => void;
  onClearFilters?: () => void;
  onSelect: (clipId: string) => void;
};

export function ClipGrid({ status, hasData, clips, resolveCameraLabel, onRetry, onClearFilters, onSelect }: ClipGridProps): JSX.Element {
  if (status === 'loading' && !hasData) {
    return <p className="py-12 text-center text-sm text-muted-foreground" role="status">이벤트를 불러오는 중입니다…</p>;
  }

  if (status === 'error' && !hasData) {
    return (
      <div className="flex flex-col items-center gap-3 py-12 text-center text-sm text-muted-foreground" role="alert">
        <p>이벤트를 불러오지 못했습니다.</p>
        <button type="button" className="dialog-secondary-action" onClick={onRetry}>다시 시도</button>
      </div>
    );
  }

  if (clips.length === 0) {
    return (
      <div className="flex flex-col items-center gap-3 py-12 text-center text-sm text-muted-foreground">
        <p>조건에 맞는 이벤트가 없습니다.</p>
        {onClearFilters ? (
          <button type="button" className="dialog-secondary-action" onClick={onClearFilters}>필터 초기화</button>
        ) : null}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {clips.map((clip) => (
        <ClipCard key={clip.id} clip={clip} cameraLabel={resolveCameraLabel(clip)} onSelect={onSelect} />
      ))}
    </div>
  );
}
