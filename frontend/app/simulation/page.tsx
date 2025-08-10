'use client';

import { AppTemplate } from '@/shared/components/layout/AppTemplate';
import { SimulationDashboard } from '@/features/simulation/components/SimulationDashboard';
import { Activity } from 'lucide-react';

export default function Page() {
  return (
    <AppTemplate
      title="最終変位・沈下予測"
      description="測定データの分析と変位予測"
      icon={<Activity className="h-8 w-8 text-blue-600" />}
      badge="難易度：中"
      badgeVariant="secondary"
      maxWidth="7xl"
    >
      <SimulationDashboard />
    </AppTemplate>
  );
} 