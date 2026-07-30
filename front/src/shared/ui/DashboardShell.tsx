import { ReactNode } from 'react';
import type { DashboardPage } from '@/app/dashboardLocation';
import { SeniorAiLabBrand } from '@/shared/ui/SeniorAiLabBrand';
import { useAuthSession } from '@/shared/ui/AuthGate';

type BackendStatusView = { label: string; className: string };
const destinations: Array<{ id: DashboardPage; label: string }> = [
  { id: 'operations', label: '관제' },
  { id: 'events', label: '이벤트 기록' },
  { id: 'cameras', label: '카메라 관리' },
  { id: 'system', label: '시스템' },
];

export function getScreenLabel(screen: DashboardPage): string {
  return destinations.find((entry) => entry.id === screen)?.label ?? screen;
}

function Navigation({ active, onChange, label }: { active: DashboardPage; onChange: (screen: DashboardPage) => void; label: string }): JSX.Element {
  return (
    <nav aria-label={label}>
      {destinations.map((entry) => (
        <button key={entry.id} type="button" aria-current={active === entry.id ? 'page' : undefined} onClick={() => onChange(entry.id)}>
          <span aria-hidden="true" className="nav-mark">{entry.label.slice(0, 1)}</span>
          <span>{entry.label}</span>
        </button>
      ))}
    </nav>
  );
}

export function DashboardShell({
  backendStatus,
  screen,
  onScreenChange,
  onAddCamera,
  children,
}: {
  apiBase?: string;
  backendStatus?: BackendStatusView;
  screen: DashboardPage;
  onScreenChange: (screen: DashboardPage) => void;
  onAddCamera?: () => void;
  children: ReactNode;
}): JSX.Element {
  const session = useAuthSession();
  return (
    <div className="dashboard-shell">
      <a className="skip-link" href="#main-content">본문으로 건너뛰기</a>
      <aside className="desktop-rail">
        <SeniorAiLabBrand />
        <Navigation active={screen} onChange={onScreenChange} label="주요 내비게이션" />
      </aside>
      <div className="dashboard-workspace">
        <header className="dashboard-topbar">
          <div className="mobile-brand"><SeniorAiLabBrand compact /></div>
          <div>
            <p className="eyebrow">운영 콘솔</p>
            <p className="shell-page-title">{getScreenLabel(screen)}</p>
          </div>
          <div className="shell-actions">
            {backendStatus ? <span className={backendStatus.className}>{backendStatus.label}</span> : null}
            {screen === 'cameras' && onAddCamera ? <button type="button" className="brand-action" onClick={onAddCamera}>카메라 추가</button> : null}
            {session ? <button type="button" onClick={session.logout}>로그아웃</button> : null}
          </div>
        </header>
        <main id="main-content" tabIndex={-1}>{children}</main>
      </div>
      <div className="mobile-tabs"><Navigation active={screen} onChange={onScreenChange} label="하단 내비게이션" /></div>
    </div>
  );
}
