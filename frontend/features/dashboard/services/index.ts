import { DashboardCard, ProjectStats } from '../types';

export const dashboardService = {
  getDashboardCards: (): DashboardCard[] => [
    {
      id: 'simulation',
      title: '最終変位・沈下予測',
      description: '測定データの分析と変位予測',
      icon: 'Activity',
      route: '/simulation',
      difficulty: 'medium',
      variant: 'default'
    },
    {
      id: 'template',
      title: 'テンプレート',
      description: '開発用のテンプレートとコンポーネント',
      icon: 'Palette',
      route: '/template',
      variant: 'outline'
    },
    {
      id: 'settings',
      title: '設定',
      description: 'アプリケーションの設定と管理',
      icon: 'Settings',
      route: '/settings',
      variant: 'outline'
    }
  ],

  getProjectStats: (): ProjectStats => ({
    completedProjects: 0,
    monthlyDesigns: 0,
    averageDesignTime: '-'
  })
}; 