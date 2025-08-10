import React from 'react';
import { Card, CardContent } from '@/components/ui/layout/card';
import { Target, Clock, DollarSign, Users, TrendingUp } from 'lucide-react';
import { StatCard as StatCardType } from '../types/common';

interface StatCardProps {
  stat: StatCardType;
}

export const StatCard: React.FC<StatCardProps> = ({ stat }) => {
  const getIcon = (iconType: string) => {
    switch (iconType) {
      case 'target':
        return <Target className="w-5 h-5 text-green-600" />;
      case 'clock':
        return <Clock className="w-5 h-5 text-blue-600" />;
      case 'dollar':
        return <DollarSign className="w-5 h-5 text-purple-600" />;
      case 'users':
        return <Users className="w-5 h-5 text-orange-600" />;
      default:
        return null;
    }
  };

  const getTrendIcon = (trend: string) => {
    return trend === 'up' ? (
      <TrendingUp className="w-4 h-4 text-green-600" />
    ) : (
      <TrendingUp className="w-4 h-4 text-red-600 rotate-180" />
    );
  };

  return (
    <Card className="bg-white border border-gray-200">
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-gray-600">{stat.title}</p>
            <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
          </div>
          <div className="flex items-center gap-2">
            {getIcon(stat.iconType)}
          </div>
        </div>
        <div className="flex items-center gap-1 mt-2">
          {getTrendIcon(stat.trend)}
          <span className={`text-sm font-medium ${
            stat.trend === 'up' ? 'text-green-600' : 'text-red-600'
          }`}>
            {stat.change}
          </span>
          <span className="text-sm text-gray-500">前期比</span>
        </div>
      </CardContent>
    </Card>
  );
}; 