'use client';
  
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Header } from '@/components/layout/header';
import { Sidebar } from '@/components/layout/sidebar';
import { Button } from '@/components/ui/forms/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/layout/card';
import { Activity, Settings, BarChart3, Brain, Home, TrendingUp } from 'lucide-react';

interface DashboardCard {
  id: string;
  title: string;
  description: string;
  icon: string;
  route: string;
  difficulty?: 'low' | 'medium' | 'high';
  variant?: 'default' | 'outline';
}

const dashboardCards: DashboardCard[] = [
  {
    id: 'measurements',
    title: 'A計測集計',
    description: '構造物の変位・沈下データの可視化と分析',
    icon: 'BarChart3',
    route: '/measurements',
    difficulty: 'low',
    variant: 'default'
  },
  {
    id: 'learning',
    title: '予測モデル作成',
    description: 'AIを活用した構造物の変位予測と学習分析',
    icon: 'Brain',
    route: '/learning',
    difficulty: 'high',
    variant: 'default'
  },
  {
    id: 'simulation',
    title: '最終変位・沈下予測',
    description: '局所変位解析とリアルタイム監視',
    icon: 'Activity',
    route: '/simulation',
    difficulty: 'medium',
    variant: 'default'
  },
  {
    id: 'settings',
    title: '設定',
    description: 'システム設定と監視パラメータの調整',
    icon: 'Settings',
    route: '/settings',
    difficulty: 'low',
    variant: 'default'
  }
];

const iconMap = {
  Activity,
  Settings,
  BarChart3,
  Brain,
  Home,
  TrendingUp
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
  Settings: 'text-orange-600',
  BarChart3: 'text-green-600',
  Brain: 'text-purple-600',
  Home: 'text-indigo-600',
  TrendingUp: 'text-red-600'
};

export default function HomePage() {
  const router = useRouter();
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  const handleMobileSidebarToggle = () => {
    setIsMobileSidebarOpen(!isMobileSidebarOpen);
  };

  const handleMobileSidebarClose = () => {
    setIsMobileSidebarOpen(false);
  };

  const handleCardClick = (route: string) => {
    router.push(route);
  };
  
  return (
    <div className="min-h-screen bg-white">
      <Sidebar 
        isMobileOpen={isMobileSidebarOpen}
        onMobileClose={handleMobileSidebarClose}
      />
      
      <div className="flex flex-col min-h-screen md:pl-64">
        <Header onMobileSidebarToggle={handleMobileSidebarToggle} />
        
        <main className="flex-1 w-full px-4 py-8 bg-white">
          <div className="max-w-7xl mx-auto">
            <div className="mb-8">
              <div className="flex items-center gap-3 mb-4">
                <Settings className="w-8 h-8 text-blue-600" />
                <div>
                  <h1 className="text-3xl font-bold text-gray-900">
                    演算工房 AI-CN
                  </h1>
                </div>
              </div>
              <p className="text-gray-600">
                構造物の変位・沈下データの監視とAIによる予測分析を行うシステムです
              </p>
            </div>

            {/* AI-A計測 Section */}
            <div className="mb-12">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">AI-A計測</h2>
              <p className="text-gray-600 mb-6">構造物の変位・沈下データの監視とAI分析</p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {dashboardCards.filter(card => ['measurements', 'learning', 'simulation'].includes(card.id)).map((card) => {
                  const IconComponent = iconMap[card.icon as keyof typeof iconMap];
                  const iconColor = iconColors[card.icon as keyof typeof iconColors] || 'text-gray-600';
                  const isClickable = !!card.route;

                  return (
                    <Card 
                      key={card.id}
                      className={`bg-white border border-gray-200 ${isClickable ? 'hover:shadow-md transition-shadow cursor-pointer' : ''}`}
                      onClick={isClickable ? () => handleCardClick(card.route) : undefined}
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
                        <Button className="w-full">
                          {card.id === 'measurements' ? 'A計測集計を開始' : 
                           card.id === 'learning' ? '予測モデル作成を実行' :
                           card.id === 'simulation' ? '最終変位・沈下予測を実行' : 'スタート'}
                        </Button>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </div>

            {/* その他 Section */}
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">その他</h2>
              <p className="text-gray-600 mb-6">システム設定とその他の機能</p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {dashboardCards.filter(card => card.id === 'settings').map((card) => {
                  const IconComponent = iconMap[card.icon as keyof typeof iconMap];
                  const iconColor = iconColors[card.icon as keyof typeof iconColors] || 'text-gray-600';
                  const isClickable = !!card.route;

                  return (
                    <Card 
                      key={card.id}
                      className={`bg-white border border-gray-200 ${isClickable ? 'hover:shadow-md transition-shadow cursor-pointer' : ''}`}
                      onClick={isClickable ? () => handleCardClick(card.route) : undefined}
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
                        <Button className="w-full">
                          {card.id === 'settings' ? '設定を開く' : 'スタート'}
                        </Button>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
} 