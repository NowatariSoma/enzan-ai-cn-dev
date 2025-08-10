'use client';

import { AppTemplate } from '@/shared/components/layout/AppTemplate';
import { Brain } from 'lucide-react';
import { LearningPage } from '@/features/learning/components/LearningPage';

export default function Page() {
  return (
    <AppTemplate
      title="予測モデル作成"
      description="AIを活用した構造物の変位予測と学習分析"
      icon={<Brain className="h-8 w-8 text-blue-600" />}
      badge="AI解析"
      badgeVariant="secondary"
      maxWidth="7xl"
    >
      <LearningPage />
    </AppTemplate>
  );
} 