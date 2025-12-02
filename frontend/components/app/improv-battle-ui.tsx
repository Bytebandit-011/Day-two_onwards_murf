'use client';

import { SessionProvider } from '@/components/app/session-provider';
import { SessionView } from '@/components/app/session-view';
import { APP_CONFIG_DEFAULTS } from '@/app-config';

export default function ImprovBattleApp() {
  // Simple config - agent name matches backend
  const appConfig = {
    ...APP_CONFIG_DEFAULTS,
    agentName: 'improv-battle-host',
    pageTitle: 'Improv Battle',
    startButtonText: 'Start Improv Battle',
    supportsChatInput: false,
    supportsVideoInput: false,
    supportsScreenShare: false,
  };

  return (
    <SessionProvider config={appConfig}>
      <SessionView />
    </SessionProvider>
  );
}
