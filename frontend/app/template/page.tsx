'use client';

import { AppTemplate } from '@/shared/components/layout/AppTemplate';
import { TemplatePage } from '@/features/template';

export default function Template() {
  return (
    <AppTemplate
      title="テンプレートレイアウト"
      description="統一されたレイアウトテンプレートのデモンストレーション"
      maxWidth="7xl"
    >
      <TemplatePage />
    </AppTemplate>
  );
} 