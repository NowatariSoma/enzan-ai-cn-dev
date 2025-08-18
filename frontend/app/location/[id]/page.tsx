'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { Header } from '@/components/layout/header';
import { Sidebar } from '@/components/layout/sidebar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/layout/card';
import { Button } from '@/components/ui/forms/button';
import { Badge } from '@/components/ui/data-display/badge';
import { Progress } from '@/components/ui/data-display/progress';
import { getLocationById } from '@/lib/data/locations';
import { 
  Activity, 
  BarChart3, 
  Brain, 
  MapPin, 
  Calendar,
  AlertTriangle,
  TrendingUp,
  Clock,
  CheckCircle,
  Settings,
  Users,
  Ruler,
  Mountain,
  Droplets,
  Wind
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';

export default function LocationDashboard() {
  const params = useParams();
  const router = useRouter();
  const locationId = params.id as string;
  const location = getLocationById(locationId);
  
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [timeRange, setTimeRange] = useState('7d'); // 7d, 30d, 90d, all

  useEffect(() => {
    if (location) {
      localStorage.setItem('selectedLocation', JSON.stringify(location));
    }
  }, [location]);

  if (!location) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">拠点が見つかりません</h2>
          <Button onClick={() => router.push('/')}>
            ダッシュボードに戻る
          </Button>
        </div>
      </div>
    );
  }

  // モックデータ生成
  const generateTimeSeriesData = () => {
    const days = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : timeRange === '90d' ? 90 : 180;
    const data = [];
    const now = new Date();
    
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      data.push({
        date: date.toLocaleDateString('ja-JP', { month: 'short', day: 'numeric' }),
        変位量: Math.sin(i * 0.3) * 5 + Math.random() * 2 - 10,
        沈下量: Math.cos(i * 0.2) * 3 + Math.random() * 1.5 - 8,
        予測値: Math.sin(i * 0.3) * 4 - 9,
      });
    }
    return data;
  };

  const timeSeriesData = generateTimeSeriesData();

  // リスク評価データ
  const riskData = [
    { subject: '地質条件', A: 65, B: 75, fullMark: 100 },
    { subject: '施工条件', A: 85, B: 90, fullMark: 100 },
    { subject: '気象影響', A: 45, B: 55, fullMark: 100 },
    { subject: '周辺環境', A: 70, B: 80, fullMark: 100 },
    { subject: '構造安定性', A: 88, B: 92, fullMark: 100 },
  ];

  // 日次進捗データ
  const dailyProgressData = [
    { name: '月', 計画: 4.5, 実績: 4.2 },
    { name: '火', 計画: 4.8, 実績: 5.1 },
    { name: '水', 計画: 4.6, 実績: 4.7 },
    { name: '木', 計画: 4.9, 実績: 4.5 },
    { name: '金', 計画: 5.2, 実績: 5.8 },
    { name: '土', 計画: 3.5, 実績: 3.2 },
    { name: '日', 計画: 0, 実績: 0 },
  ];

  const statusColors = {
    active: 'bg-emerald-500 text-white',
    monitoring: 'bg-blue-500 text-white',
    completed: 'bg-gray-400 text-white',
    planning: 'bg-amber-500 text-white'
  };

  const statusLabels = {
    active: '稼働中',
    monitoring: '監視中',
    completed: '完了',
    planning: '計画中'
  };

  const alertColors = {
    normal: 'border-emerald-200 bg-emerald-50 text-emerald-700',
    warning: 'border-amber-200 bg-amber-50 text-amber-700',
    danger: 'border-rose-200 bg-rose-50 text-rose-700'
  };

  const alertIcons = {
    normal: <CheckCircle className="h-5 w-5 text-emerald-500" />,
    warning: <AlertTriangle className="h-5 w-5 text-amber-500" />,
    danger: <AlertTriangle className="h-5 w-5 text-rose-500" />
  };

  const handleMobileSidebarToggle = () => {
    setIsMobileSidebarOpen(!isMobileSidebarOpen);
  };

  const handleMobileSidebarClose = () => {
    setIsMobileSidebarOpen(false);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar 
        isMobileOpen={isMobileSidebarOpen}
        onMobileClose={handleMobileSidebarClose}
      />
      
      <div className="flex flex-col min-h-screen md:pl-64">
        <Header onMobileSidebarToggle={handleMobileSidebarToggle} />
        
        <main className="flex-1 w-full px-4 py-6">
          <div className="max-w-7xl mx-auto">
            {/* ヘッダー */}
            <div className="mb-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <div className="flex items-center gap-3 mb-2">
                    <MapPin className="h-6 w-6 text-blue-600" />
                    <h1 className="text-2xl font-bold text-gray-900">
                      {location.tunnelName}
                    </h1>
                    <Badge className={statusColors[location.status]}>
                      {statusLabels[location.status]}
                    </Badge>
                  </div>
                  <p className="text-gray-600">{location.description}</p>
                  <p className="text-sm text-gray-500">{location.region} / {location.prefecture}</p>
                </div>
                
                {/* 時間範囲選択 */}
                <div className="flex gap-2">
                  {['7d', '30d', '90d', 'all'].map((range) => (
                    <Button
                      key={range}
                      variant={timeRange === range ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setTimeRange(range)}
                    >
                      {range === '7d' ? '7日' : 
                       range === '30d' ? '30日' : 
                       range === '90d' ? '90日' : '全期間'}
                    </Button>
                  ))}
                </div>
              </div>

              {/* アラートバー */}
              <div className={`flex items-center justify-between p-3 rounded-lg border ${alertColors[location.alertLevel]}`}>
                <div className="flex items-center gap-2">
                  {alertIcons[location.alertLevel]}
                  <span className="font-medium">
                    システムステータス: {location.alertLevel === 'normal' ? '正常' : 
                                       location.alertLevel === 'warning' ? '注意' : '警告'}
                  </span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <Clock className="h-4 w-4" />
                  <span>最終更新: {new Date(location.lastUpdated).toLocaleString('ja-JP')}</span>
                </div>
              </div>
            </div>

            {/* AI-A計測アクションカード */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <Card className="bg-white hover:shadow-lg transition-shadow cursor-pointer border-2 hover:border-green-500"
                    onClick={() => router.push(`/measurements?location=${location.id}`)}>
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <BarChart3 className="h-5 w-5 text-green-600" />
                    A計測集計
                  </CardTitle>
                  <CardDescription className="text-sm">
                    詳細な変位・沈下データの分析
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Button className="w-full bg-green-600 hover:bg-green-700">
                    <Activity className="h-4 w-4 mr-2" />
                    詳細を見る
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-white hover:shadow-lg transition-shadow cursor-pointer border-2 hover:border-blue-500"
                    onClick={() => router.push(`/simulation?location=${location.id}`)}>
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <Activity className="h-5 w-5 text-blue-600" />
                    最終変位・沈下予測
                  </CardTitle>
                  <CardDescription className="text-sm">
                    シミュレーションによる予測分析
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Button className="w-full bg-blue-600 hover:bg-blue-700">
                    <TrendingUp className="h-4 w-4 mr-2" />
                    予測実行
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-white hover:shadow-lg transition-shadow cursor-pointer border-2 hover:border-purple-500"
                    onClick={() => router.push(`/learning?location=${location.id}`)}>
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <Brain className="h-5 w-5 text-purple-600" />
                    予測モデル作成
                  </CardTitle>
                  <CardDescription className="text-sm">
                    AIによる予測モデルの構築と学習
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Button className="w-full bg-purple-600 hover:bg-purple-700">
                    <Brain className="h-4 w-4 mr-2" />
                    モデル作成
                  </Button>
                </CardContent>
              </Card>
            </div>

            {/* メトリクスカード */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <Card className="bg-white">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">工事進捗</p>
                      <p className="text-2xl font-bold text-gray-900">{location.progress}%</p>
                      <Progress value={location.progress} className="mt-2" />
                    </div>
                    <TrendingUp className="h-8 w-8 text-blue-500 opacity-20" />
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-white">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">トンネル全長</p>
                      <p className="text-2xl font-bold text-gray-900">{location.totalLength}m</p>
                      <p className="text-xs text-gray-500 mt-1">
                        完成: {Math.floor(location.totalLength * location.progress / 100)}m
                      </p>
                    </div>
                    <Ruler className="h-8 w-8 text-green-500 opacity-20" />
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-white">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">計測ポイント</p>
                      <p className="text-2xl font-bold text-gray-900">{location.measurementCount}</p>
                      <p className="text-xs text-gray-500 mt-1">アクティブ</p>
                    </div>
                    <Activity className="h-8 w-8 text-purple-500 opacity-20" />
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-white">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">本日の進捗</p>
                      <p className="text-2xl font-bold text-gray-900">5.2m</p>
                      <p className="text-xs text-green-600 mt-1">+15% 計画比</p>
                    </div>
                    <Mountain className="h-8 w-8 text-orange-500 opacity-20" />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* グラフセクション */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              {/* 変位・沈下トレンド */}
              <Card className="bg-white">
                <CardHeader>
                  <CardTitle>変位・沈下トレンド</CardTitle>
                  <CardDescription>
                    過去{timeRange === '7d' ? '7日間' : timeRange === '30d' ? '30日間' : timeRange === '90d' ? '90日間' : '全期間'}のデータ
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={timeSeriesData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="変位量" 
                        stroke="#3B82F6" 
                        strokeWidth={2}
                        dot={{ r: 3 }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="沈下量" 
                        stroke="#EF4444" 
                        strokeWidth={2}
                        dot={{ r: 3 }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="予測値" 
                        stroke="#10B981" 
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* リスク評価 */}
              <Card className="bg-white">
                <CardHeader>
                  <CardTitle>リスク評価</CardTitle>
                  <CardDescription>各種リスクファクターの評価</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={riskData}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="subject" />
                      <PolarRadiusAxis angle={90} domain={[0, 100]} />
                      <Radar 
                        name="現在値" 
                        dataKey="A" 
                        stroke="#3B82F6" 
                        fill="#3B82F6" 
                        fillOpacity={0.6} 
                      />
                      <Radar 
                        name="目標値" 
                        dataKey="B" 
                        stroke="#10B981" 
                        fill="#10B981" 
                        fillOpacity={0.3} 
                      />
                      <Legend />
                    </RadarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* 日次進捗 */}
              <Card className="bg-white">
                <CardHeader>
                  <CardTitle>日次進捗</CardTitle>
                  <CardDescription>今週の掘削進捗（m/日）</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={dailyProgressData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="計画" fill="#94A3B8" />
                      <Bar dataKey="実績" fill="#3B82F6" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* 環境データ */}
              <Card className="bg-white">
                <CardHeader>
                  <CardTitle>環境モニタリング</CardTitle>
                  <CardDescription>リアルタイム環境データ</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Droplets className="h-5 w-5 text-blue-600" />
                      <span className="text-sm font-medium">湧水量</span>
                    </div>
                    <span className="text-lg font-bold text-blue-900">125 L/min</span>
                  </div>
                  
                  <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Wind className="h-5 w-5 text-green-600" />
                      <span className="text-sm font-medium">換気状態</span>
                    </div>
                    <span className="text-lg font-bold text-green-900">良好</span>
                  </div>
                  
                  <div className="flex items-center justify-between p-3 bg-amber-50 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Mountain className="h-5 w-5 text-amber-600" />
                      <span className="text-sm font-medium">地山等級</span>
                    </div>
                    <span className="text-lg font-bold text-amber-900">CII</span>
                  </div>
                  
                  <div className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
                    <div className="flex items-center gap-2">
                      <Activity className="h-5 w-5 text-purple-600" />
                      <span className="text-sm font-medium">振動レベル</span>
                    </div>
                    <span className="text-lg font-bold text-purple-900">42 dB</span>
                  </div>
                </CardContent>
              </Card>
            </div>

          </div>
        </main>
      </div>
    </div>
  );
}