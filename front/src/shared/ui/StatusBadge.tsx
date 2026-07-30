import type { CameraStatus, SystemSnapshot } from '@/shared/api/client';

const cameraStatusMeta: Record<CameraStatus, { label: string; className: string }> = {
  online: {
    label: '온라인',
    className: 'bg-status-stableBg text-status-stable ring-status-stable',
  },
  offline: {
    label: '오프라인',
    className: 'bg-status-dangerBg text-status-danger ring-status-danger',
  },
  starting: {
    label: '시작 중',
    className: 'bg-surface2 text-ink-soft ring-border',
  },
  unknown: {
    label: '확인 중',
    className: 'bg-surface2 text-ink-soft ring-border',
  },
};

export function getCameraStatusMeta(status: CameraStatus): { label: string; className: string } {
  return cameraStatusMeta[status];
}

type StatusBadgeProps = {
  status: CameraStatus;
};

export function StatusBadge({ status }: StatusBadgeProps): JSX.Element {
  const meta = getCameraStatusMeta(status);

  return (
    <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-bold ring-1 ${meta.className}`}>
      <span className="mr-1.5 h-2 w-2 rounded-full bg-current" />
      {meta.label}
    </span>
  );
}

export function getBackendStatus(system: SystemSnapshot | null): { label: string; className: string } {
  if (!system) {
    return {
      label: '백엔드 확인 중',
      className: 'bg-surface2 text-ink-soft ring-border',
    };
  }

  if (!system.backend.configured) {
    return {
      label: '백엔드 미설정',
      className: 'bg-status-cautionBg text-status-caution ring-status-caution',
    };
  }

  if (system.backend.reachable === true) {
    return {
      label: '백엔드 연결됨',
      className: 'bg-status-stableBg text-status-stable ring-status-stable',
    };
  }

  if (system.backend.reachable === false) {
    return {
      label: '백엔드 연결 실패',
      className: 'bg-status-dangerBg text-status-danger ring-status-danger',
    };
  }

  return {
    label: '백엔드 대기 중',
    className: 'bg-surface2 text-ink-soft ring-border',
  };
}
