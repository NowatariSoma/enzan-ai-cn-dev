'use client';
  
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Header } from '@/components/layout/header';
import { Sidebar } from '@/components/layout/sidebar';
import { Button } from '@/components/ui/forms/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/layout/card';
import { LocationCard } from '@/components/dashboard/LocationCard';
import { Badge } from '@/components/ui/data-display/badge';
import { locations, getLocationsByRegion, getLocationStats, getAlertStats } from '@/lib/data/locations';
import { Activity, Settings, BarChart3, Brain, Home, TrendingUp, MapPin, AlertTriangle, CheckCircle, Clock } from 'lucide-react';

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
  const [selectedRegion, setSelectedRegion] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');

  const locationsByRegion = getLocationsByRegion();
  const locationStats = getLocationStats();
  const alertStats = getAlertStats();

  const handleMobileSidebarToggle = () => {
    setIsMobileSidebarOpen(!isMobileSidebarOpen);
  };

  const handleMobileSidebarClose = () => {
    setIsMobileSidebarOpen(false);
  };

  const handleCardClick = (route: string) => {
    router.push(route);
  };

  const handleLocationSelect = (location: typeof locations[0]) => {
    localStorage.setItem('selectedLocation', JSON.stringify(location));
  };

  // フィルタリングされた拠点リスト
  const filteredLocations = locations.filter(location => {
    const regionMatch = selectedRegion === 'all' || location.region === selectedRegion;
    const statusMatch = selectedStatus === 'all' || location.status === selectedStatus;
    return regionMatch && statusMatch;
  });
  
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
            {/* ヘッダー */}
            <div className="mb-8">
              <div className="flex items-center gap-3 mb-4">
                <MapPin className="w-8 h-8 text-blue-600" />
                <div>
                  <h1 className="text-3xl font-bold text-gray-900">
                    演算工房 AI-CN 拠点管理
                  </h1>
                </div>
              </div>
              <p className="text-gray-600">
                全国のトンネル工事現場における変位・沈下データの監視とAI予測分析
              </p>
            </div>

            {/* 統計サマリー */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
              <Card className="bg-white">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">稼働中</p>
                      <p className="text-2xl font-bold text-green-600">{locationStats.active}</p>
                    </div>
                    <Activity className="h-8 w-8 text-green-600 opacity-20" />
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-white">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">監視中</p>
                      <p className="text-2xl font-bold text-blue-600">{locationStats.monitoring}</p>
                    </div>
                    <Clock className="h-8 w-8 text-blue-600 opacity-20" />
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-white">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">警告</p>
                      <p className="text-2xl font-bold text-red-600">{alertStats.danger}</p>
                    </div>
                    <AlertTriangle className="h-8 w-8 text-red-600 opacity-20" />
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-white">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">完了</p>
                      <p className="text-2xl font-bold text-gray-600">{locationStats.completed}</p>
                    </div>
                    <CheckCircle className="h-8 w-8 text-gray-600 opacity-20" />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* フィルター */}
            <div className="flex flex-wrap gap-4 mb-6">
              <div className="flex gap-2">
                <Button
                  variant={selectedRegion === 'all' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedRegion('all')}
                >
                  全地域
                </Button>
                {Object.keys(locationsByRegion).map(region => (
                  <Button
                    key={region}
                    variant={selectedRegion === region ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setSelectedRegion(region)}
                  >
                    {region}
                  </Button>
                ))}
              </div>
              
              <div className="flex gap-2">
                <Button
                  variant={selectedStatus === 'all' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedStatus('all')}
                >
                  全状態
                </Button>
                <Button
                  variant={selectedStatus === 'active' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedStatus('active')}
                >
                  稼働中
                </Button>
                <Button
                  variant={selectedStatus === 'monitoring' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedStatus('monitoring')}
                >
                  監視中
                </Button>
                <Button
                  variant={selectedStatus === 'completed' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedStatus('completed')}
                >
                  完了
                </Button>
              </div>
            </div>

            {/* 拠点一覧 */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">拠点一覧 - AI-A計測管理</h2>
              
              {filteredLocations.length === 0 ? (
                <Card className="bg-white">
                  <CardContent className="p-8 text-center">
                    <p className="text-gray-500">該当する拠点が見つかりません</p>
                  </CardContent>
                </Card>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {filteredLocations.map((location) => (
                    <LocationCard
                      key={location.id}
                      location={location}
                      onSelect={handleLocationSelect}
                    />
                  ))}
                </div>
              )}
            </div>

            {/* その他 Section */}
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">システム管理</h2>
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