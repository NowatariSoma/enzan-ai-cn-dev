'use client';

import { useState, useMemo, useEffect, useCallback } from 'react';
import { useSearchParams } from 'next/navigation';
import { useHeatmap } from './useHeatmap';

interface WholeAnalysisResult {
  status: string;
  message: string;
  training_metrics?: Record<string, any>;
  scatter_data?: Record<string, any>;
  feature_importance?: Record<string, any>;
  model_files_saved?: boolean;
}

export function useLearning() {
  const searchParams = useSearchParams();
  const locationParam = searchParams?.get('location');
  const [folderName, setFolderName] = useState<string>("01-hokkaido-akan");
  const [model, setModel] = useState("Random Forest");
  const [predictionTD, setPredictionTD] = useState(500);
  const [maxDistance, setMaxDistance] = useState(100);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [dataType, setDataType] = useState<'settlement' | 'convergence'>('settlement');
  const [analysisData, setAnalysisData] = useState<WholeAnalysisResult | null>(null);

  // useHeatmapフックを使用
  const { heatmapData, features, fetchHeatmapData } = useHeatmap();

  // locationパラメータを直接フォルダ名として使用
  useEffect(() => {
    if (locationParam) {
      setFolderName(locationParam);
      console.log('Learning: Using folder name from URL parameter:', locationParam);
    }
  }, [locationParam]);

  // フォルダが変更されたときにヒートマップデータを取得
  useEffect(() => {
    fetchHeatmapData(folderName, maxDistance);
  }, [folderName, maxDistance, fetchHeatmapData]);

  // Generate mock data for charts
  const generateChartData = () => {
    const data = [];
    for (let i = 0; i <= 50; i++) {
      const distanceFromFace = i * 2;
      const noise = (Math.random() - 0.5) * 0.2;
      
      data.push({
        distanceFromFace,
        変位量A: Math.sin(distanceFromFace * 0.1) * 0.5 + noise,
        変位量B: Math.cos(distanceFromFace * 0.08) * 0.3 + noise * 0.5,
        変位量C: Math.sin(distanceFromFace * 0.12) * 0.4 + noise * 0.3,
        変位量A_prediction: Math.cos(distanceFromFace * 0.1) * 0.6 + noise * 0.4,
        変位量B_prediction: Math.sin(distanceFromFace * 0.09) * 0.35 + noise * 0.6,
        変位量C_prediction: Math.cos(distanceFromFace * 0.11) * 0.45 + noise * 0.5,
      });
    }
    return data;
  };

  // Transform scatter data from new API format (settlement/convergence separated)
  const transformScatterData = (scatterSubData: any) => {
    if (!scatterSubData || !scatterSubData.train_actual || !scatterSubData.train_predictions) return [];
    
    return scatterSubData.train_actual.map((actualValue: number, index: number) => ({
      actual: actualValue,
      predicted: scatterSubData.train_predictions[index] || 0,
    }));
  };
  
  const transformValidationScatterData = (scatterSubData: any) => {
    if (!scatterSubData || !scatterSubData.validate_actual || !scatterSubData.validate_predictions) return [];
    
    return scatterSubData.validate_actual.map((actualValue: number, index: number) => ({
      actual: actualValue,
      predicted: scatterSubData.validate_predictions[index] || 0,
    }));
  };


  const chartData = useMemo(() => generateChartData(), []);
  
  // Settlement scatter data (A)
  const trainScatterDataSettlement = useMemo(() => {
    if (analysisData?.scatter_data?.settlement) {
      return transformScatterData(analysisData.scatter_data.settlement);
    }
    return [];
  }, [analysisData]);

  const validationScatterDataSettlement = useMemo(() => {
    if (analysisData?.scatter_data?.settlement) {
      return transformValidationScatterData(analysisData.scatter_data.settlement);
    }
    return [];
  }, [analysisData]);

  // Convergence scatter data (B)
  const trainScatterDataConvergence = useMemo(() => {
    if (analysisData?.scatter_data?.convergence) {
      return transformScatterData(analysisData.scatter_data.convergence);
    }
    return [];
  }, [analysisData]);

  const validationScatterDataConvergence = useMemo(() => {
    if (analysisData?.scatter_data?.convergence) {
      return transformValidationScatterData(analysisData.scatter_data.convergence);
    }
    return [];
  }, [analysisData]);

  const featureImportanceData = useMemo(() => {
    if (analysisData?.feature_importance) {
      const featureImportance = analysisData.feature_importance;
      
      // Check if it's the new format with feature_names and importance_values arrays
      if (featureImportance.feature_names && featureImportance.importance_values) {
        return featureImportance.feature_names
          .map((feature: string, index: number) => ({
            feature,
            importance: Number(featureImportance.importance_values[index] || 0)
          }))
          .sort((a: any, b: any) => b.importance - a.importance)
          .slice(0, 10); // Top 10 features
      }
      
      // Fallback to old format
      if (typeof featureImportance === 'object') {
        return Object.entries(featureImportance)
          .map(([feature, importance]) => ({ feature, importance: Number(importance) }))
          .sort((a, b) => b.importance - a.importance)
          .slice(0, 10);
      }
    }
    return []; // Return empty array instead of mock data
  }, [analysisData]);

  // Map settlement and convergence data to A and B charts
  const trainScatterDataA = trainScatterDataSettlement;
  const trainScatterDataB = trainScatterDataConvergence;
  const validationScatterDataA = validationScatterDataSettlement;
  const validationScatterDataB = validationScatterDataConvergence;
  const featureImportanceA = featureImportanceData;
  const featureImportanceB = featureImportanceData;
  
  // 実際のヒートマップデータを使用
  const heatmapDataA = { data: heatmapData, features };
  const heatmapDataB = { data: heatmapData, features };

  // Metrics from analyze-whole endpoint  
  // Get settlement metrics directly from scatter_data.settlement.metrics
  const getSettlementMetrics = () => {
    if (analysisData?.scatter_data?.settlement?.metrics) {
      const metrics = analysisData.scatter_data.settlement.metrics;
      // Use 'final' metrics if available, otherwise use 'current'
      return metrics.final || metrics.current || { mse_train: 0, r2_train: 0, mse_validate: 0, r2_validate: 0 };
    }
    return { mse_train: 0, r2_train: 0, mse_validate: 0, r2_validate: 0 };
  };

  // Get convergence metrics directly from scatter_data.convergence.metrics
  const getConvergenceMetrics = () => {
    if (analysisData?.scatter_data?.convergence?.metrics) {
      const metrics = analysisData.scatter_data.convergence.metrics;
      // Use 'final' metrics if available, otherwise use 'current'
      return metrics.final || metrics.current || { mse_train: 0, r2_train: 0, mse_validate: 0, r2_validate: 0 };
    }
    return { mse_train: 0, r2_train: 0, mse_validate: 0, r2_validate: 0 };
  };
  
  const settlementMetrics = getSettlementMetrics();
  const convergenceMetrics = getConvergenceMetrics();
  
  // Separate metrics for settlement (A) and convergence (B)
  const trainRSquaredA = settlementMetrics.r2_train;
  const trainRSquaredB = convergenceMetrics.r2_train;
  const validationRSquaredA = settlementMetrics.r2_validate;
  const validationRSquaredB = convergenceMetrics.r2_validate;
  
  const trainMSEA = settlementMetrics.mse_train;
  const trainMSEB = convergenceMetrics.mse_train;
  const validationMSEA = settlementMetrics.mse_validate;
  const validationMSEB = convergenceMetrics.mse_validate;

  // Fetch data from analyze-whole endpoint
  const fetchAnalysisData = useCallback(async () => {
    setIsAnalyzing(true);
    console.log('Fetching analyze-whole data with params:', {
      model_name: model,
      folder_name: folderName,
      max_distance_from_face: maxDistance,
      td: predictionTD,
    });

    try {
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
      
      // AbortControllerを使用してタイムアウトを5分に設定
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000); // 5分
      
      const response = await fetch(`${API_BASE_URL}/displacement-analysis/analyze-whole`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_name: model,
          folder_name: folderName,
          max_distance_from_face: maxDistance,
          td: predictionTD,
        }),
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

      console.log('Response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('API Error:', errorText);
        throw new Error(`Failed to fetch analyze-whole data: ${response.status} ${errorText}`);
      }

      const data: WholeAnalysisResult = await response.json();
      console.log('Received data:', {
        status: data.status,
        message: data.message,
        training_metrics: data.training_metrics,
        scatter_data_keys: Object.keys(data.scatter_data || {}),
        feature_importance_keys: Object.keys(data.feature_importance || {}),
        model_files_saved: data.model_files_saved,
      });

      setAnalysisData(data);
      
      // Also fetch heatmap data
      fetchHeatmapData(folderName, maxDistance);
    } catch (error) {
      console.error('Error fetching analyze-whole data:', error);
      // Set error state or show user-friendly error
      alert(`API接続エラー: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsAnalyzing(false);
    }
  }, [model, folderName, maxDistance, predictionTD, fetchHeatmapData]);

  const handleAnalyze = () => {
    fetchAnalysisData();
  };

  return {
    folderName,
    setFolderName,
    model,
    setModel,
    dataType,
    setDataType,
    predictionTD,
    setPredictionTD,
    maxDistance,
    setMaxDistance,
    isAnalyzing,
    chartData,
    trainScatterDataA,
    trainScatterDataB,
    validationScatterDataA,
    validationScatterDataB,
    featureImportanceA,
    featureImportanceB,
    heatmapDataA,
    heatmapDataB,
    trainRSquaredA,
    trainRSquaredB,
    validationRSquaredA,
    validationRSquaredB,
    trainMSEA,
    trainMSEB,
    validationMSEA,
    validationMSEB,
    handleAnalyze,
    analysisData,
  };
}