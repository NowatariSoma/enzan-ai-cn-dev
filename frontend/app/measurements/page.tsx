'use client';

// Force dynamic rendering
export const dynamic = 'force-dynamic';

import { AppTemplate } from '@/shared/components/layout/AppTemplate';
import { BarChart3 } from 'lucide-react';
import { MeasurementsPage } from '@/features/measurements/components/MeasurementsPage';

export default function Page() {
  return (
    <AppTemplate
      title="A計測集計"
      description="構造物の変位・沈下データの可視化と分析"
      icon={<BarChart3 className="h-8 w-8 text-blue-600" />}
      badge="リアルタイム監視"
      badgeVariant="secondary"
      maxWidth="7xl"
    >
      <MeasurementsPage />
    </AppTemplate>
  );
} 