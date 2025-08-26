// API client for locations endpoints

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

// フロントエンド用インターフェース（既存のモックデータと互換性を保つ）
export interface AvailableFeatures {
  aiMeasurement: boolean;
  measurement: boolean;
  simulation: boolean;
  modelCreation: boolean;
  realTimeMonitoring?: boolean;
  riskAnalysis?: boolean;
  reportGeneration?: boolean;
  dataExport?: boolean;
  userManagement?: boolean;
  locationManagement?: boolean;
  [key: string]: boolean | undefined;
}

export interface Location {
  id: string;
  location_id?: string; // バックエンドのlocation_id
  name: string;
  region: string;
  prefecture: string;
  tunnelName: string;
  description: string;
  folderName: string;
  status: 'active' | 'monitoring' | 'completed' | 'planning';
  startDate: string;
  totalLength: number;
  progress: number;
  lastUpdated: string;
  measurementCount: number;
  alertLevel: 'normal' | 'warning' | 'danger';
  coordinates?: {
    lat: number;
    lng: number;
  };
  availableFeatures: AvailableFeatures;
}

// バックエンドレスポンス用インターフェース
export interface LocationAPIResponse {
  id: number;
  location_id: string;
  name: string;
  description: string;
  address: string;
  region: string;
  prefecture: string;
  tunnel_name: string;
  folder_name: string;
  status: 'active' | 'monitoring' | 'completed' | 'planning';
  start_date: string;
  total_length: number;
  progress: string; // Decimal field comes as string
  measurement_count: number;
  alert_level: 'normal' | 'warning' | 'danger';
  coordinates: {
    lat: number;
    lng: number;
  } | null;
  available_features: AvailableFeatures;
  created_at: string;
  updated_at: string;
}

export class LocationsAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async getLocations(): Promise<Location[]> {
    const response = await fetch(`${this.baseUrl}/locations/`);
    if (!response.ok) {
      throw new Error(`Failed to fetch locations: ${response.statusText}`);
    }
    const data: LocationAPIResponse[] = await response.json();
    return data.map(this.transformLocationResponse);
  }

  async getLocation(id: string): Promise<Location> {
    const response = await fetch(`${this.baseUrl}/locations/${id}/`);
    if (!response.ok) {
      throw new Error(`Failed to fetch location: ${response.statusText}`);
    }
    const data: LocationAPIResponse = await response.json();
    return this.transformLocationResponse(data);
  }

  // バックエンドのレスポンスをフロントエンド用の形式に変換
  private transformLocationResponse(apiResponse: LocationAPIResponse): Location {
    return {
      id: apiResponse.location_id, // フロントエンド用のIDはlocation_idを使用
      location_id: apiResponse.location_id,
      name: apiResponse.name,
      region: apiResponse.region || '',
      prefecture: apiResponse.prefecture || '',
      tunnelName: apiResponse.tunnel_name || '',
      description: apiResponse.description || '',
      folderName: apiResponse.folder_name || '',
      status: apiResponse.status,
      startDate: apiResponse.start_date || '',
      totalLength: apiResponse.total_length || 0,
      progress: parseFloat(apiResponse.progress) || 0,
      lastUpdated: apiResponse.updated_at,
      measurementCount: apiResponse.measurement_count || 0,
      alertLevel: apiResponse.alert_level,
      coordinates: apiResponse.coordinates,
      availableFeatures: apiResponse.available_features || {
        aiMeasurement: false,
        measurement: false,
        simulation: false,
        modelCreation: false,
      }
    };
  }
}

// Export a default instance
export const locationsAPI = new LocationsAPI();

// 地域ごとの拠点をグループ化
export function getLocationsByRegion(locations: Location[]): Record<string, Location[]> {
  return locations.reduce((acc, location) => {
    if (!acc[location.region]) {
      acc[location.region] = [];
    }
    acc[location.region].push(location);
    return acc;
  }, {} as Record<string, Location[]>);
}

// IDから拠点を取得
export function getLocationById(locations: Location[], id: string): Location | undefined {
  return locations.find(loc => loc.id === id);
}

// フォルダ名から拠点を取得
export function getLocationByFolderName(locations: Location[], folderName: string): Location | undefined {
  return locations.find(loc => loc.folderName === folderName);
}

// ステータスごとの拠点数を取得
export function getLocationStats(locations: Location[]) {
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
export function getAlertStats(locations: Location[]) {
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