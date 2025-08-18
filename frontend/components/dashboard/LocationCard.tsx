'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/layout/card';
import { Button } from '@/components/ui/forms/button';
import { Progress } from '@/components/ui/data-display/progress';
import { Badge } from '@/components/ui/data-display/badge';
import { Location } from '@/lib/data/locations';
import { MapPin, Activity, AlertTriangle, Calendar, TrendingUp } from 'lucide-react';
import { useRouter } from 'next/navigation';

interface LocationCardProps {
  location: Location;
  onSelect: (location: Location) => void;
}

export function LocationCard({ location, onSelect }: LocationCardProps) {
  const router = useRouter();

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
    normal: 'text-emerald-500',
    warning: 'text-amber-500',
    danger: 'text-rose-500'
  };

  const AlertIcon = location.alertLevel === 'danger' ? AlertTriangle : Activity;

  const handleViewDashboard = () => {
    // 拠点専用ダッシュボードへ遷移
    localStorage.setItem('selectedLocation', JSON.stringify(location));
    router.push(`/location/${location.id}`);
  };

  const handleAMeasurement = () => {
    // 拠点を選択してA計測画面へ遷移
    localStorage.setItem('selectedLocation', JSON.stringify(location));
    router.push(`/measurements?location=${location.id}`);
  };

  const handlePrediction = () => {
    // 拠点を選択して予測画面へ遷移
    localStorage.setItem('selectedLocation', JSON.stringify(location));
    router.push(`/learning?location=${location.id}`);
  };

  const handleSimulation = () => {
    // 拠点を選択してシミュレーション画面へ遷移
    localStorage.setItem('selectedLocation', JSON.stringify(location));
    router.push(`/simulation?location=${location.id}`);
  };

  return (
    <Card 
      className="bg-white border border-gray-200 hover:shadow-lg hover:border-blue-300 transition-all duration-200 overflow-hidden cursor-pointer"
      onClick={handleViewDashboard}>
      <CardHeader className="pb-4 bg-white border-b border-gray-100">
        <div className="flex justify-between items-start mb-2">
          <div className="flex items-center gap-2">
            <MapPin className="h-5 w-5 text-blue-600" />
            <span className="text-sm font-medium text-gray-700">{location.region} / {location.prefecture}</span>
          </div>
          <Badge className={statusColors[location.status]}>
            {statusLabels[location.status]}
          </Badge>
        </div>
        <CardTitle className="text-xl font-bold text-gray-900">
          {location.tunnelName}
        </CardTitle>
        <CardDescription className="text-gray-600">
          {location.description}
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* 進捗状況 */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">工事進捗</span>
            <span className="font-semibold">{location.progress}%</span>
          </div>
          <Progress value={location.progress} className="h-2" />
          <div className="flex justify-between text-xs text-gray-500">
            <span>全長: {location.totalLength}m</span>
            <span>計測点: {location.measurementCount}箇所</span>
          </div>
        </div>

        {/* アラートステータス */}
        <div className={`flex items-center justify-between p-3 rounded-lg ${
          location.alertLevel === 'normal' ? 'bg-emerald-50 border border-emerald-200' :
          location.alertLevel === 'warning' ? 'bg-amber-50 border border-amber-200' :
          'bg-rose-50 border border-rose-200'
        }`}>
          <div className="flex items-center gap-2">
            <AlertIcon className={`h-5 w-5 ${alertColors[location.alertLevel]}`} />
            <span className={`text-sm font-medium ${
              location.alertLevel === 'normal' ? 'text-emerald-700' :
              location.alertLevel === 'warning' ? 'text-amber-700' :
              'text-rose-700'
            }`}>
              {location.alertLevel === 'normal' ? '正常' : 
               location.alertLevel === 'warning' ? '注意' : '警告'}
            </span>
          </div>
          <div className="flex items-center gap-1 text-xs text-gray-500">
            <Calendar className="h-3 w-3" />
            <span>更新: {new Date(location.lastUpdated).toLocaleDateString('ja-JP')}</span>
          </div>
        </div>

        {/* アクションボタン */}
        <div className="grid grid-cols-1 gap-2">
          <Button 
            onClick={(e) => {
              e.stopPropagation();
              handleViewDashboard();
            }}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white"
            disabled={location.status === 'planning'}
          >
            <Activity className="h-4 w-4 mr-2" />
            拠点ダッシュボードを開く
          </Button>
          
          <div className="grid grid-cols-3 gap-2">
            <Button 
              onClick={(e) => {
                e.stopPropagation();
                handleAMeasurement();
              }}
              variant="outline"
              size="sm"
              className="text-xs"
              disabled={location.status === 'planning'}
            >
              A計測
            </Button>
            <Button 
              onClick={(e) => {
                e.stopPropagation();
                handleSimulation();
              }}
              variant="outline"
              size="sm"
              className="text-xs"
              disabled={location.status === 'planning'}
            >
              最終予測
            </Button>
            <Button 
              onClick={(e) => {
                e.stopPropagation();
                handlePrediction();
              }}
              variant="outline"
              size="sm"
              className="text-xs"
              disabled={location.status === 'planning'}
            >
              モデル作成
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}