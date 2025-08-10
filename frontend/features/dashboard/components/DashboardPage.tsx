'use client';

import { AppTemplate } from '@/shared/components/layout/AppTemplate';
import { DashboardCard } from './DashboardCard';
import { RecentProjects } from './RecentProjects';
import { useDashboard } from '../hooks';

export function DashboardPage() {
  const { cards, stats, isLoading } = useDashboard();

  if (isLoading) {
    return (
      <AppTemplate
        title="tomosigoto 自動製造プランナー"
        description="お客様の要望から最適な製造装置の設計・選定・レイアウトを自動化するシステムです"
      >
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="bg-gray-200 rounded-lg h-48"></div>
            </div>
          ))}
        </div>
      </AppTemplate>
    );
  }

  return (
    <AppTemplate
      title="tomosigoto 自動製造プランナー"
      description="お客様の要望から最適な製造装置の設計・選定・レイアウトを自動化するシステムです"
    >
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {cards.map((card) => (
          <DashboardCard key={card.id} card={card} />
        ))}
      </div>

      <div className="mt-8">
        <RecentProjects />
      </div>
    </AppTemplate>
  );
} 