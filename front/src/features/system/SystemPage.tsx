import { useStatusResource, useSystemResource } from '@/shared/api/usePollingResource';
import { SystemPanel } from '@/shared/ui/SystemPanels';

export function SystemPage({ active }: { active: boolean }): JSX.Element {
  const systemResource = useSystemResource(active);
  const statusResource = useStatusResource(active);
  return <SystemPanel systemResource={systemResource} statusResource={statusResource} />;
}
