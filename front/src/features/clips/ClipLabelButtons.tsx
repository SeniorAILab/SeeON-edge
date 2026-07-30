import { useState } from 'react';
import { labelClip, type Clip, type ClipLabel } from '@/shared/api/client';

const LABELS: Array<{ value: ClipLabel; className: string }> = [
  { value: 'TRUE_POSITIVE', className: 'bg-status-stableBg text-status-stable hover:bg-status-stableBg' },
  { value: 'FALSE_POSITIVE', className: 'bg-status-dangerBg text-status-danger hover:bg-status-dangerBg' },
  { value: 'UNREVIEWED', className: 'bg-surface2 text-ink-soft hover:bg-surface2' },
];

type ClipLabelButtonsProps = {
  clip: Clip;
  onChanged: (clip: Clip) => void;
};

function positiveLabel(eventType?: string): string {
  const normalized = eventType?.trim().toLowerCase().replace(/[\s_]+/g, '-');
  if (normalized === 'bed-exit') return '실제 침대 이탈';
  if (normalized === 'fall') return '실제 낙상';
  return '실제 이벤트';
}

export function koreanClipLabel(
  label: ClipLabel | null,
  reviewState: Clip['reviewState'] = 'confirmed',
  eventType?: string,
): string {
  if (reviewState === 'unknown') return '라벨 정보 없음';
  if (label === 'TRUE_POSITIVE') return positiveLabel(eventType);
  if (label === 'FALSE_POSITIVE') return '오탐';
  return '미검토';
}

export function ClipLabelButtons({ clip, onChanged }: ClipLabelButtonsProps): JSX.Element {
  const [busyLabel, setBusyLabel] = useState<ClipLabel | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const selectedLabel = clip.label ?? (clip.reviewState === 'confirmed' ? 'UNREVIEWED' : null);

  async function handleClick(nextLabel: ClipLabel): Promise<void> {
    setBusyLabel(nextLabel);
    setMessage(null);
    try {
      const updated = await labelClip(clip, nextLabel);
      onChanged(updated);
      setMessage(`${koreanClipLabel(nextLabel, 'confirmed', clip.event_type)} 라벨을 저장했습니다.`);
    } catch {
      setMessage('라벨 저장에 실패했습니다.');
    } finally {
      setBusyLabel(null);
    }
  }

  return (
    <div>
      <p className="mb-2 text-xs font-bold text-ink-soft">{koreanClipLabel(clip.label, clip.reviewState, clip.event_type)}</p>
      <div className="flex flex-wrap gap-2">
        {LABELS.map((entry) => {
          const selected = selectedLabel === entry.value;
          return (
            <button
              key={entry.value}
              type="button"
              onClick={() => void handleClick(entry.value)}
              disabled={busyLabel !== null}
              aria-pressed={selected}
              className={`rounded-full px-3 py-2 text-xs font-black disabled:cursor-not-allowed disabled:opacity-60 ${entry.className} ${
                selected ? 'ring-2 ring-brand' : ''
              }`}
            >
              {busyLabel === entry.value ? '저장 중...' : koreanClipLabel(entry.value, 'confirmed', clip.event_type)}
            </button>
          );
        })}
      </div>
      {message ? (
        <p className="mt-2 text-xs font-bold text-ink-soft" role="status">
          {message}
        </p>
      ) : null}
    </div>
  );
}
