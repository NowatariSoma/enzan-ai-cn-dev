// 拠点データの定義

// 利用可能な機能の定義
export interface AvailableFeatures {
  aiMeasurement: boolean;           // AI-A計測集計
  measurement: boolean;              // A計測集計
  simulation: boolean;               // 最終変位・沈下予測  
  modelCreation: boolean;            // 予測モデル作成
  realTimeMonitoring?: boolean;      // リアルタイム監視（将来的な機能）
  riskAnalysis?: boolean;            // リスク分析（将来的な機能）
  reportGeneration?: boolean;        // レポート生成（将来的な機能）
  dataExport?: boolean;              // データエクスポート（将来的な機能）
  [key: string]: boolean | undefined; // 拡張用
}

export interface Location {
  id: string;
  name: string;
  region: string;
  prefecture: string;
  tunnelName: string;
  description: string;
  folderName: string; // バックエンドのフォルダ名
  status: 'active' | 'monitoring' | 'completed' | 'planning';
  startDate: string;
  totalLength: number; // トンネル全長（m）
  progress: number; // 進捗率（%）
  lastUpdated: string;
  measurementCount: number; // 計測ポイント数
  alertLevel: 'normal' | 'warning' | 'danger';
  coordinates?: {
    lat: number;
    lng: number;
  };
  availableFeatures: AvailableFeatures; // 利用可能な機能
}

export const locations: Location[] = [
  {
    id: '01-hokkaido-akan',
    name: '北海道阿寒',
    region: '北海道',
    prefecture: '北海道',
    tunnelName: '阿寒トンネル',
    description: '国道240号線の山岳トンネル工事',
    folderName: '01-hokkaido-akan',
    status: 'active',
    startDate: '2024-04-01',
    totalLength: 1157,
    progress: 78.5,
    lastUpdated: '2025-08-18',
    measurementCount: 64,
    alertLevel: 'normal',
    coordinates: {
      lat: 43.4500,
      lng: 144.0167
    },
    availableFeatures: {
      aiMeasurement: true,
      measurement: true,
      simulation: true,
      modelCreation: true,
      realTimeMonitoring: true,
      riskAnalysis: true,
      reportGeneration: true,
      dataExport: true
    }
  },
  {
    id: '02-hokkaido-atsuga',
    name: '北海道厚賀',
    region: '北海道',
    prefecture: '北海道',
    tunnelName: '厚賀トンネル',
    description: '日高自動車道延伸工事',
    folderName: '01-hokkaido-atsuga',
    status: 'active',
    startDate: '2024-06-15',
    totalLength: 2450,
    progress: 45.2,
    lastUpdated: '2025-08-18',
    measurementCount: 48,
    alertLevel: 'warning',
    coordinates: {
      lat: 42.1083,
      lng: 142.5639
    },
    availableFeatures: {
      aiMeasurement: true,
      measurement: true,
      simulation: true,
      modelCreation: false,  // モデル作成は未対応
      realTimeMonitoring: true,
      riskAnalysis: false
    }
  },
  {
    id: '03-tohoku-zao',
    name: '東北蔵王',
    region: '東北',
    prefecture: '宮城県',
    tunnelName: '蔵王トンネル',
    description: '東北自動車道バイパス工事',
    folderName: '03-tohoku-zao',
    status: 'monitoring',
    startDate: '2023-09-01',
    totalLength: 3200,
    progress: 92.3,
    lastUpdated: '2025-08-17',
    measurementCount: 72,
    alertLevel: 'normal',
    coordinates: {
      lat: 38.1000,
      lng: 140.5667
    },
    availableFeatures: {
      aiMeasurement: true,
      measurement: true,
      simulation: true,
      modelCreation: true,
      reportGeneration: true
    }
  },
  {
    id: '04-kanto-hakone',
    name: '関東箱根',
    region: '関東',
    prefecture: '神奈川県',
    tunnelName: '新箱根トンネル',
    description: '国道1号線バイパストンネル',
    folderName: '04-kanto-hakone',
    status: 'active',
    startDate: '2024-02-01',
    totalLength: 1850,
    progress: 63.7,
    lastUpdated: '2025-08-18',
    measurementCount: 56,
    alertLevel: 'normal',
    coordinates: {
      lat: 35.2333,
      lng: 139.0167
    },
    availableFeatures: {
      aiMeasurement: true,
      measurement: true,
      simulation: true,
      modelCreation: true,
      realTimeMonitoring: true,
      dataExport: true
    }
  },
  {
    id: '05-chubu-fuji',
    name: '中部富士',
    region: '中部',
    prefecture: '静岡県',
    tunnelName: '富士山麓トンネル',
    description: '新東名高速道路延伸工事',
    folderName: '05-chubu-fuji',
    status: 'active',
    startDate: '2024-01-15',
    totalLength: 4500,
    progress: 34.8,
    lastUpdated: '2025-08-18',
    measurementCount: 88,
    alertLevel: 'warning',
    coordinates: {
      lat: 35.3606,
      lng: 138.7274
    },
    availableFeatures: {
      aiMeasurement: true,
      measurement: true,
      simulation: false,  // シミュレーション準備中
      modelCreation: true,
      riskAnalysis: true
    }
  },
  {
    id: '06-kansai-rokko',
    name: '関西六甲',
    region: '関西',
    prefecture: '兵庫県',
    tunnelName: '新六甲トンネル',
    description: '阪神高速道路延伸工事',
    folderName: '06-kansai-rokko',
    status: 'completed',
    startDate: '2023-03-01',
    totalLength: 2750,
    progress: 100.0,
    lastUpdated: '2025-08-10',
    measurementCount: 62,
    alertLevel: 'normal',
    coordinates: {
      lat: 34.7333,
      lng: 135.2000
    },
    availableFeatures: {
      aiMeasurement: false,  // 完了済みのため無効
      measurement: false,
      simulation: false,
      modelCreation: false,
      reportGeneration: true,  // レポート閲覧のみ可能
      dataExport: true
    }
  },
  {
    id: '07-chugoku-okayama',
    name: '中国岡山',
    region: '中国',
    prefecture: '岡山県',
    tunnelName: '吉備トンネル',
    description: '山陽自動車道改良工事',
    folderName: '07-chugoku-okayama',
    status: 'planning',
    startDate: '2025-10-01',
    totalLength: 1650,
    progress: 0,
    lastUpdated: '2025-08-15',
    measurementCount: 0,
    alertLevel: 'normal',
    coordinates: {
      lat: 34.6617,
      lng: 133.9350
    },
    availableFeatures: {
      aiMeasurement: false,  // 計画中のため無効
      measurement: false,
      simulation: false,
      modelCreation: false
    }
  },
  {
    id: '08-shikoku-kochi',
    name: '四国高知',
    region: '四国',
    prefecture: '高知県',
    tunnelName: '四万十トンネル',
    description: '高知自動車道延伸工事',
    folderName: '08-shikoku-kochi',
    status: 'active',
    startDate: '2024-05-01',
    totalLength: 2100,
    progress: 51.4,
    lastUpdated: '2025-08-18',
    measurementCount: 44,
    alertLevel: 'normal',
    coordinates: {
      lat: 33.5597,
      lng: 133.5311
    },
    availableFeatures: {
      aiMeasurement: true,
      measurement: true,
      simulation: true,
      modelCreation: true
    }
  },
  {
    id: '09-kyushu-aso',
    name: '九州阿蘇',
    region: '九州',
    prefecture: '熊本県',
    tunnelName: '阿蘇山トンネル',
    description: '九州横断自動車道工事',
    folderName: '09-kyushu-aso',
    status: 'active',
    startDate: '2024-03-15',
    totalLength: 3850,
    progress: 28.9,
    lastUpdated: '2025-08-18',
    measurementCount: 68,
    alertLevel: 'danger',
    coordinates: {
      lat: 32.8847,
      lng: 131.1042
    },
    availableFeatures: {
      aiMeasurement: true,
      measurement: true,
      simulation: true,  // 緊急対応モード
      modelCreation: false,  // リスクが高いため制限
      realTimeMonitoring: true,
      riskAnalysis: true
    }
  },
  {
    id: '10-okinawa-naha',
    name: '沖縄那覇',
    region: '沖縄',
    prefecture: '沖縄県',
    tunnelName: '首里トンネル',
    description: '那覇都市モノレール延伸工事',
    folderName: '10-okinawa-naha',
    status: 'monitoring',
    startDate: '2023-11-01',
    totalLength: 850,
    progress: 88.5,
    lastUpdated: '2025-08-17',
    measurementCount: 32,
    alertLevel: 'normal',
    coordinates: {
      lat: 26.2167,
      lng: 127.6792
    },
    availableFeatures: {
      aiMeasurement: true,
      measurement: true,
      simulation: false,
      modelCreation: true,
      reportGeneration: true
    }
  }
];

// 地域ごとの拠点をグループ化
export function getLocationsByRegion(): Record<string, Location[]> {
  return locations.reduce((acc, location) => {
    if (!acc[location.region]) {
      acc[location.region] = [];
    }
    acc[location.region].push(location);
    return acc;
  }, {} as Record<string, Location[]>);
}

// IDから拠点を取得
export function getLocationById(id: string): Location | undefined {
  return locations.find(loc => loc.id === id);
}

// フォルダ名から拠点を取得
export function getLocationByFolderName(folderName: string): Location | undefined {
  return locations.find(loc => loc.folderName === folderName);
}

// ステータスごとの拠点数を取得
export function getLocationStats() {
  const stats = {
    active: 0,
    monitoring: 0,
    completed: 0,
    planning: 0,
    total: locations.length
  };
  
  locations.forEach(loc => {
    stats[loc.status]++;
  });
  
  return stats;
}

// アラートレベルごとの拠点数を取得
export function getAlertStats() {
  const stats = {
    normal: 0,
    warning: 0,
    danger: 0
  };
  
  locations.forEach(loc => {
    stats[loc.alertLevel]++;
  });
  
  return stats;
}