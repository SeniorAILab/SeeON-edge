import { useEffect, useMemo, useState } from 'react';
import type { Camera, Clip } from '@/shared/api/client';
import { ClipLabelButtons, koreanClipLabel } from '@/features/clips/ClipLabelButtons';
import {
  eventHistoryLocationUpdate,
  eventLabel,
  filterHistoricalClips,
  formatEventTime,
  groupHistoricalClips,
  resolveClipCamera,
  type EventHistoryLocation,
} from '@/features/events/cameraEventLogic';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';

type ResourceStatus = 'idle' | 'loading' | 'success' | 'error';

export type EventHistoryPageProps = {
  cameras: Camera[];
  clips: Clip[];
  status: ResourceStatus;
  lastSuccessAt: number | null;
  refreshing: boolean;
  location: EventHistoryLocation;
  onNavigate: (update: Partial<Record<'floor' | 'room' | 'camera' | 'event' | 'clip', string | null>>) => void;
  onRetry: () => void;
  onClipChanged: (clip: Clip) => void;
};

function unique(values: Array<string | null>): string[] {
  return [...new Set(values.filter((value): value is string => value !== null))].sort((left, right) => left.localeCompare(right, 'ko'));
}

function cameraName(clip: Clip, cameras: readonly Camera[]): string {
  return resolveClipCamera(clip, cameras)?.label ?? (clip.camera_id ? '카메라 정보 없음' : clip.camera_label || '카메라 정보 없음');
}

function locationName(clip: Clip, cameras: readonly Camera[]): string {
  const camera = resolveClipCamera(clip, cameras);
  if (!camera) return '위치 정보 없음';
  return [camera.floor_name, camera.space_name ?? camera.space_id].filter(Boolean).join(' · ') || '위치 정보 없음';
}

function mediaLabel(clip: Clip, cameras: readonly Camera[]): string {
  return `${cameraName(clip, cameras)} ${eventLabel(clip.event_type)} ${formatEventTime(clip.created_at)} 증거 영상`;
}

export function EventHistoryPage({
  cameras,
  clips,
  status,
  lastSuccessAt,
  refreshing,
  location,
  onNavigate,
  onRetry,
  onClipChanged,
}: EventHistoryPageProps): JSX.Element {
  const [labelOverlays, setLabelOverlays] = useState<Record<string, Pick<Clip, 'label' | 'reviewer' | 'reviewed_at' | 'reviewState'>>>({});
  const [videoPlaybackFailed, setVideoPlaybackFailed] = useState(false);
  const visibleClips = useMemo(
    () => filterHistoricalClips(cameras, clips, location).map((clip) => ({ ...clip, ...labelOverlays[clip.id] })),
    [cameras, clips, labelOverlays, location],
  );
  const selectedClip = visibleClips.find((clip) => clip.id === location.clip) ?? null;
  const locationGroups = useMemo(() => groupHistoricalClips(cameras, visibleClips), [cameras, visibleClips]);
  const floorOptions = unique(cameras.map((camera) => camera.floor_name));
  const roomOptions = unique(cameras.filter((camera) => !location.floor || camera.floor_name === location.floor).map((camera) => camera.space_id));
  const cameraOptions = cameras.filter((camera) => (!location.floor || camera.floor_name === location.floor) && (!location.room || camera.space_id === location.room));
  const eventOptions = unique(filterHistoricalClips(cameras, clips, {
    floor: location.floor,
    room: location.room,
    camera: location.camera,
  }).map((clip) => clip.event_type));

  useEffect(() => {
    setVideoPlaybackFailed(false);
  }, [selectedClip?.id, selectedClip?.video_path]);

  function navigate(key: 'floor' | 'room' | 'camera' | 'event' | 'clip', value: string | null): void {
    const next = eventHistoryLocationUpdate(location, key, value);
    onNavigate(Object.fromEntries(
      (['floor', 'room', 'camera', 'event', 'clip'] as const)
        .filter((entry) => next[entry] !== location[entry])
        .map((entry) => [entry, next[entry] ?? null]),
    ));
  }

  function handleClipChanged(updated: Clip): void {
    setLabelOverlays((current) => ({
      ...current,
      [updated.id]: {
        label: updated.label,
        reviewer: updated.reviewer,
        reviewed_at: updated.reviewed_at,
        reviewState: updated.reviewState,
      },
    }));
    onClipChanged(updated);
  }

  const firstLoading = (status === 'idle' || status === 'loading') && lastSuccessAt === null;
  const staleError = status === 'error' && lastSuccessAt !== null;
  const retainedResultIsEmpty = staleError && visibleClips.length === 0;
  let errorMessage = '이벤트 기록을 불러오지 못했습니다.';
  if (staleError) {
    errorMessage = retainedResultIsEmpty
      ? '마지막 확인 당시 저장된 기록이 없습니다.'
      : '목록 갱신이 지연되어 최근 기록을 표시하고 있습니다.';
  }

  return (
    <section aria-labelledby="event-history-heading" className="space-y-5">
      <header>
        <p className="text-sm font-bold text-ink-soft">저장된 증거 클립</p>
        <h1 id="event-history-heading" data-dialog-focus-fallback tabIndex={-1} className="text-xl font-bold text-ink">이벤트 기록</h1>
      </header>

      <div aria-label="이벤트 기록 필터" className="flex flex-wrap gap-3">
        <label>층<select className="event-filter-control" aria-label="층 필터" value={location.floor ?? ''} onChange={(event) => navigate('floor', event.target.value || null)}><option value="">전체</option>{floorOptions.map((value) => <option key={value}>{value}</option>)}</select></label>
        <label>공간<select className="event-filter-control" aria-label="공간 필터" value={location.room ?? ''} onChange={(event) => navigate('room', event.target.value || null)}><option value="">전체</option>{roomOptions.map((value) => <option key={value} value={value}>{cameras.find((camera) => camera.space_id === value)?.space_name ?? value}</option>)}</select></label>
        <label>카메라<select className="event-filter-control" aria-label="카메라 필터" value={location.camera ?? ''} onChange={(event) => navigate('camera', event.target.value || null)}><option value="">전체</option>{cameraOptions.map((camera) => <option key={camera.id} value={camera.id}>{camera.label}</option>)}</select></label>
        <label>이벤트<select className="event-filter-control" aria-label="이벤트 필터" value={location.event ?? ''} onChange={(event) => navigate('event', event.target.value || null)}><option value="">전체</option>{eventOptions.map((value) => <option key={value} value={value}>{eventLabel(value)}</option>)}</select></label>
      </div>

      {firstLoading ? <p role="status">이벤트 기록을 불러오는 중입니다.</p> : null}
      {refreshing && lastSuccessAt !== null ? <p role="status" aria-live="polite">이벤트 기록을 새로 확인하고 있습니다.</p> : null}
      {status === 'error' ? (
        <div
          role={staleError ? 'status' : 'alert'}
          aria-live={staleError ? 'polite' : undefined}
          data-testid={retainedResultIsEmpty ? 'event-history-empty' : undefined}
        >
          <p>{errorMessage}</p>
          {lastSuccessAt !== null ? <p>마지막 확인 {new Date(lastSuccessAt).toLocaleString('ko-KR')}</p> : null}
          <button type="button" onClick={onRetry}>다시 시도</button>
        </div>
      ) : null}

      {!firstLoading && status === 'success' && visibleClips.length === 0
        ? <p data-testid="event-history-empty">저장된 이벤트 기록이 없습니다.</p>
        : null}
      {locationGroups.length > 0 ? (
        <div className="space-y-5" data-testid="event-history-list">
          {locationGroups.map((group) => (
            <section key={group.key} data-testid="event-location-group" aria-label={`${group.floor} ${group.room}`}>
              <h2>{group.floor} · {group.room}</h2>
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {group.clips.map((clip) => (
                  <article key={clip.id} data-testid="event-clip" className="rounded-2xl border border-border bg-surface p-4">
                    <p className="font-bold text-ink">{eventLabel(clip.event_type)}</p>
                    <p className="text-sm text-ink-soft">{cameraName(clip, cameras)}</p>
                    <p className="text-sm text-ink-soft">{locationName(clip, cameras)}</p>
                    <p className="text-sm text-ink-soft">{formatEventTime(clip.created_at)}</p>
                    <p className="text-sm text-ink-soft">{koreanClipLabel(clip.label, clip.reviewState, clip.event_type)}</p>
                    <button type="button" onClick={() => navigate('clip', clip.id)} aria-label={`${mediaLabel(clip, cameras)} 열기`}>증거 보기</button>
                  </article>
                ))}
              </div>
            </section>
          ))}
        </div>
      ) : null}

      <AccessibleDialog open={selectedClip !== null} title="이벤트 증거" onClose={() => navigate('clip', null)} initialFocus="heading">
        {selectedClip ? (
          <article>
            <p>{eventLabel(selectedClip.event_type)} · {cameraName(selectedClip, cameras)}</p>
            <p>{locationName(selectedClip, cameras)} · {formatEventTime(selectedClip.created_at)}</p>
            <div className="event-media-frame">
              {selectedClip.video_available && !videoPlaybackFailed ? (
                <video src={selectedClip.video_path} preload="metadata" controls onError={() => setVideoPlaybackFailed(true)} aria-label={mediaLabel(selectedClip, cameras)} className="h-full w-full object-contain" />
              ) : (
                <p role="status" aria-label={mediaLabel(selectedClip, cameras)} className="event-media-unavailable">영상을 불러올 수 없음</p>
              )}
            </div>
            <ClipLabelButtons clip={selectedClip} onChanged={handleClipChanged} />
            {selectedClip.reviewState === 'confirmed' && selectedClip.reviewer ? <p>검토자 {selectedClip.reviewer} · {formatEventTime(selectedClip.reviewed_at ?? null)}</p> : null}
            <button type="button" onClick={() => navigate('clip', null)}>닫기</button>
          </article>
        ) : null}
      </AccessibleDialog>
    </section>
  );
}
