'use client';

import { useRouter } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/layout/card';
import { Button } from '@/components/ui/forms/button';
import { 
  Activity, 
  Palette, 
  Settings 
} from 'lucide-react';
import { DashboardCard as DashboardCardType } from '../types';

const iconMap = {
  Activity,
  Palette,
  Settings
};

const difficultyColors = {
  low: 'text-green-600',
  medium: 'text-yellow-600',
  high: 'text-red-600'
};

const difficultyLabels = {
  low: '低',
  medium: '中',
  high: '高'
};

const iconColors = {
  Activity: 'text-blue-600',
  Palette: 'text-purple-600',
  Settings: 'text-orange-600'
};

interface DashboardCardProps {
  card: DashboardCardType;
}

export function DashboardCard({ card }: DashboardCardProps) {
  const router = useRouter();
  const IconComponent = iconMap[card.icon as keyof typeof iconMap];
  const iconColor = iconColors[card.icon as keyof typeof iconColors] || 'text-gray-600';

  const handleClick = () => {
    if (card.route) {
      router.push(card.route);
    }
  };

  const isClickable = !!card.route;

  return (
    <Card 
      className={`bg-white border border-gray-200 ${isClickable ? 'hover-card cursor-pointer' : ''}`}
      onClick={isClickable ? handleClick : undefined}
    >
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {IconComponent && <IconComponent className={`h-5 w-5 ${iconColor}`} />}
          {card.title}
        </CardTitle>
        <CardDescription>
          {card.description}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {card.variant === 'default' ? (
          <Button className="w-full">
            {card.id === 'simulation' ? '最終変位・沈下予測を実行' : 'スタート'}
          </Button>
        ) : (
          <Button variant="outline" className="w-full">
            {card.id === 'template' ? 'テンプレートを開く' : 
             card.id === 'settings' ? '設定を開く' : 'スタート'}
          </Button>
        )}
        
        {card.difficulty && (
          <div className="mt-3 text-sm text-gray-500">
            <div className="flex justify-between">
              <span>難易度：</span>
              <span className={`font-medium ${difficultyColors[card.difficulty]}`}>
                {difficultyLabels[card.difficulty]}
              </span>
            </div>
          </div>
        )}
        
      </CardContent>
    </Card>
  );
} 