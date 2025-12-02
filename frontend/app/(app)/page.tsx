'use client';

import { App } from '@/components/app/app';
import { APP_CONFIG_DEFAULTS } from '@/app-config';

export default function Page() {
  return <App appConfig={APP_CONFIG_DEFAULTS} />;
}
