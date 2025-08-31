// API client for measurements endpoints

const API_BASE_URL = '/api/v1'; // Use Next.js rewrites for API proxy

export interface TimeSeriesDataPoint {
  td: number;
  series3m: number;
  series5m: number;
  series10m: number;
  series20m: number;
  series50m: number;
  series100m: number;
}

export interface DisplacementSeriesResponse {
  data: TimeSeriesDataPoint[];
  unit: string;
  measurement_type: string;
}

export interface SettlementSeriesResponse {
  data: TimeSeriesDataPoint[];
  unit: string;
  measurement_type: string;
}

export interface DistributionDataPoint {
  range: string;
  series3m: number;
  series5m: number;
  series10m: number;
  series20m: number;
  series50m: number;
  series100m: number;
  dummyBase?: number;  // ダミーデータ（基準値用）
}

export interface DisplacementDistributionResponse {
  data: DistributionDataPoint[];
  bin_size: number;
  measurement_type: string;
}

export interface SettlementDistributionResponse {
  data: DistributionDataPoint[];
  bin_size: number;
  measurement_type: string;
}

export interface TunnelScatterPoint {
  x: number;
  y: number;
  depth: number;
  color: string;
}

export interface TunnelScatterResponse {
  data: TunnelScatterPoint[];
  x_label: string;
  y_label: string;
  color_scale: string;
}

export interface TDDataPoint {
  td: number;
  settlements: number[];
  convergences: number[];
}

export interface DistanceDataResponse {
  dct_df_td: { [key: string]: TDDataPoint[] };
  settlements: { [key: string]: number[] };
  convergences: { [key: string]: number[] };
  settlements_columns: string[];
  convergences_columns: string[];
  distances: string[];
  df_all?: any[];  // 全計測データ
}

export interface ScatterPlotPoint {
  x: number;  // 切羽からの距離
  y: number;  // 計測経過日数
  value: number;  // 変位量または沈下量
  td?: number;  // TD値
}

export interface ScatterPlotDataResponse {
  data: ScatterPlotPoint[];
  label: string;
  x_label: string;
  y_label: string;
  color_range: {
    min: number;
    max: number;
  };
  plot_type: 'convergences' | 'settlements';
}

export class MeasurementsAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async getDisplacementSeries(numPoints: number = 100, folderName: string = '01-hokkaido-akan'): Promise<DisplacementSeriesResponse> {
    const response = await fetch(`${this.baseUrl}/measurements/displacement-series?num_points=${numPoints}&folder_name=${folderName}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch displacement series: ${response.statusText}`);
    }
    return response.json();
  }

  async getSettlementSeries(numPoints: number = 100, folderName: string = '01-hokkaido-akan'): Promise<SettlementSeriesResponse> {
    const response = await fetch(`${this.baseUrl}/measurements/settlement-series?num_points=${numPoints}&folder_name=${folderName}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch settlement series: ${response.statusText}`);
    }
    return response.json();
  }

  async getDisplacementDistribution(folderName: string = '01-hokkaido-akan'): Promise<DisplacementDistributionResponse> {
    const response = await fetch(`${this.baseUrl}/measurements/displacement-distribution?folder_name=${folderName}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch displacement distribution: ${response.statusText}`);
    }
    return response.json();
  }

  async getSettlementDistribution(folderName: string = '01-hokkaido-akan'): Promise<SettlementDistributionResponse> {
    const response = await fetch(`${this.baseUrl}/measurements/settlement-distribution?folder_name=${folderName}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch settlement distribution: ${response.statusText}`);
    }
    return response.json();
  }

  async getTunnelScatter(numPoints: number = 200, folderName: string = '01-hokkaido-akan'): Promise<TunnelScatterResponse> {
    const response = await fetch(`${this.baseUrl}/measurements/tunnel-scatter?num_points=${numPoints}&folder_name=${folderName}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch tunnel scatter data: ${response.statusText}`);
    }
    return response.json();
  }

  async getDistanceData(
    folderName: string = '01-hokkaido-akan',
    maxDistanceFromFace: number = 100
  ): Promise<DistanceDataResponse> {
    const response = await fetch(
      `${this.baseUrl}/measurements/distance-data?folder_name=${folderName}&max_distance_from_face=${maxDistanceFromFace}`
    );
    if (!response.ok) {
      throw new Error(`Failed to fetch distance data: ${response.statusText}`);
    }
    return response.json();
  }

  async getScatterPlotData(
    plotType: 'convergences' | 'settlements' = 'convergences',
    folderName: string = '01-hokkaido-akan',
    maxDistanceFromFace: number = 100
  ): Promise<ScatterPlotDataResponse> {
    const response = await fetch(
      `${this.baseUrl}/measurements/scatter-plot-data?plot_type=${plotType}&folder_name=${folderName}&max_distance_from_face=${maxDistanceFromFace}`
    );
    if (!response.ok) {
      throw new Error(`Failed to fetch scatter plot data: ${response.statusText}`);
    }
    return response.json();
  }
}

// Export a default instance
export const measurementsAPI = new MeasurementsAPI();