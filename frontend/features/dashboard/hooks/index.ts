import { useState, useEffect } from 'react';
import { dashboardService } from '../services';
import { DashboardCard, ProjectStats } from '../types';

export const useDashboard = () => {
  const [cards, setCards] = useState<DashboardCard[]>([]);
  const [stats, setStats] = useState<ProjectStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadDashboardData = async () => {
      try {
        setIsLoading(true);
        const dashboardCards = dashboardService.getDashboardCards();
        const projectStats = dashboardService.getProjectStats();
        
        setCards(dashboardCards);
        setStats(projectStats);
      } catch (error) {
        console.error('Failed to load dashboard data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadDashboardData();
  }, []);

  return {
    cards,
    stats,
    isLoading
  };
}; 