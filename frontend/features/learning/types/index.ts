// Learning feature types
export interface ChartData {
  time: number;
  measurement: number;
  prediction: number;
  upperBound: number;
  lowerBound: number;
}

export interface ChartLine {
  dataKey: string;
  stroke: string;
  name: string;
  strokeDasharray?: string;
}

export interface PredictionData {
  step: number;
  days: number;
  prediction1: string;
  prediction2: string;
  prediction3: string;
  prediction4: string;
  prediction5: string;
}

export interface ScatterData {
  actual: number;
  predicted: number;
}

export interface FeatureImportanceData {
  feature: string;
  importance: number;
}

export interface LearningParams {
  folder: string;
  model: string;
  predictionTD: number;
  maxDistance: number;
} 