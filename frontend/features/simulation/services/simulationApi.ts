const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export interface LocalDisplacementRequest {
  folder_name: string;
  ameasure_file: string;
  distance_from_face: number;
  daily_advance: number;
  max_distance_from_face?: number;
}

export interface SimulationDataPoint {
  distanceFromFace: number;
  [key: string]: number; // For dynamic displacement columns
}

export interface LocalDisplacementResponse {
  folder_name: string;
  cycle_no: number;
  td: number;
  distance_from_face: number;
  daily_advance: number;
  prediction_charts: {
    settlement: string;
    convergence: string;
  };
  simulation_charts: {
    settlement: string;
    convergence: string;
  };
  simulation_csv: string;
  simulation_data: Array<{
    distance_from_face: number;
    [key: string]: number;
  }>;
  prediction_data?: Array<{
    distance_from_face: number;
    [key: string]: number;
  }>;
  table_data?: Array<{
    distance_from_face: number;
    [key: string]: number;
  }>;
  timestamp: string;
}

export interface FoldersResponse {
  folders: string[];
}

export interface MeasurementFilesResponse {
  measurement_files: string[];
}

export const simulationApi = {
  async getFolders(): Promise<FoldersResponse> {
    const response = await fetch(`${API_BASE_URL}/simulation/folders`);
    if (!response.ok) {
      throw new Error(`Failed to fetch folders: ${response.statusText}`);
    }
    return response.json();
  },

  async getMeasurementFiles(folderName: string): Promise<MeasurementFilesResponse> {
    const response = await fetch(`${API_BASE_URL}/simulation/measurements/${encodeURIComponent(folderName)}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch measurement files: ${response.statusText}`);
    }
    return response.json();
  },

  async analyzeLocalDisplacement(params: LocalDisplacementRequest): Promise<LocalDisplacementResponse> {
    const response = await fetch(`${API_BASE_URL}/simulation/local-displacement`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        folder_name: params.folder_name,
        ameasure_file: params.ameasure_file,
        distance_from_face: params.distance_from_face,
        daily_advance: params.daily_advance,
        max_distance_from_face: params.max_distance_from_face || 200.0,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to analyze displacement: ${error}`);
    }

    return response.json();
  },

  async getChartImage(path: string): Promise<string> {
    // Convert the server path to a URL that can be used to fetch the image
    const filename = path.split('/').pop();
    return `${API_BASE_URL}/static/${filename}`;
  }
};