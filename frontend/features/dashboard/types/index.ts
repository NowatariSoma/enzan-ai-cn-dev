export interface DashboardCard {
  id: string;
  title: string;
  description: string;
  icon: string;
  route: string;
  difficulty?: 'low' | 'medium' | 'high';
  variant?: 'default' | 'outline';
  features?: string[];
}

export interface ProjectStats {
  completedProjects: number;
  monthlyDesigns: number;
  averageDesignTime: string;
}

export interface DashboardProps {
  cards?: DashboardCard[];
  stats?: ProjectStats;
} 