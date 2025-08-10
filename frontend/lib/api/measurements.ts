// API client for measurements endpoints

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

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

export class MeasurementsAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async getDisplacementSeries(numPoints: number = 100): Promise<DisplacementSeriesResponse> {
    const response = await fetch(`${this.baseUrl}/measurements/displacement-series?num_points=${numPoints}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch displacement series: ${response.statusText}`);
    }
    return response.json();
  }

  async getSettlementSeries(numPoints: number = 100): Promise<SettlementSeriesResponse> {
    const response = await fetch(`${this.baseUrl}/measurements/settlement-series?num_points=${numPoints}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch settlement series: ${response.statusText}`);
    }
    return response.json();
  }

  async getDisplacementDistribution(): Promise<DisplacementDistributionResponse> {
    const response = await fetch(`${this.baseUrl}/measurements/displacement-distribution`);
    if (!response.ok) {
      throw new Error(`Failed to fetch displacement distribution: ${response.statusText}`);
    }
    return response.json();
  }

  async getSettlementDistribution(): Promise<SettlementDistributionResponse> {
    const response = await fetch(`${this.baseUrl}/measurements/settlement-distribution`);
    if (!response.ok) {
      throw new Error(`Failed to fetch settlement distribution: ${response.statusText}`);
    }
    return response.json();
  }

  async getTunnelScatter(numPoints: number = 200): Promise<TunnelScatterResponse> {
    const response = await fetch(`${this.baseUrl}/measurements/tunnel-scatter?num_points=${numPoints}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch tunnel scatter data: ${response.statusText}`);
    }
    return response.json();
  }
}

// Export a default instance
export const measurementsAPI = new MeasurementsAPI();