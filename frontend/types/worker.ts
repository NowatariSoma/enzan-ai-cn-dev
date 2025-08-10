export interface WorkerCount {
  red: number;
  other: number;
  no_helmet: number;
  total: number;
  timestamp: string;
}

// Camera API response types
export interface CameraStatus {
  message: string;
}

export interface CameraData {
  message: {
    timestamp: string;
    model: string;
    persons: Array<{
      red_helmet?: { conf: number; xyxyn: number[] };
      other_helmet?: { conf: number; xyxyn: number[] };
      no_helmet?: { conf: number; xyxyn: number[] };
    }>;
    data: Array<{
      timestamp: string;
      name: string;
      value: number;
    }>;
  };
}

// Helper type for display labels
export interface HelmetTypeLabels {
  red_helmet: string;
  other_helmet: string;
  no_helmet: string;
}

export const HELMET_LABELS: HelmetTypeLabels = {
  red_helmet: '管理者',
  other_helmet: '作業員',
  no_helmet: 'ヘルメットなし'
};

export interface HistoricalData {
  id: string;
  counts: WorkerCount;
  imageUrl: string;
  createdAt: string;
}

export interface AggregatedData {
  timeRange: string;
  averageCounts: WorkerCount;
  maxCounts: WorkerCount;
  minCounts: WorkerCount;
  totalReadings: number;
}

export interface User {
  id: string;
  username: string;
  email: string;
  role: string;
}

export interface ProjectSettings {
  id: string;
  name: string;
  location: string;
  cameraUrl: string;
  helmetColors: string[];
}