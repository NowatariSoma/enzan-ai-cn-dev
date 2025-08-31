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

  // Transform scatter data from new API format
  const transformScatterData = (scatterData: any) => {
    if (!scatterData || !scatterData.train_actual || !scatterData.train_predictions) return [];
    
    return scatterData.train_actual.map((actualValue: number, index: number) => ({
      actual: actualValue,
      predicted: scatterData.train_predictions[index] || 0,
    }));
  };
  
  const transformValidationScatterData = (scatterData: any) => {
    if (!scatterData || !scatterData.validate_actual || !scatterData.validate_predictions) return [];
    
    return scatterData.validate_actual.map((actualValue: number, index: number) => ({
      actual: actualValue,
      predicted: scatterData.validate_predictions[index] || 0,
    }));
  };


  const chartData = useMemo(() => generateChartData(), []);
  
  // Use actual data from analyze-whole endpoint if available, otherwise use mock data
  const trainScatterData = useMemo(() => {
    if (analysisData?.scatter_data) {
      return transformScatterData(analysisData.scatter_data);
    }
    return [];
  }, [analysisData]);

  const validationScatterData = useMemo(() => {
    if (analysisData?.scatter_data) {
      return transformValidationScatterData(analysisData.scatter_data);
    }
    return [];
  }, [analysisData]);

  const featureImportanceData = useMemo(() => {
    if (analysisData?.feature_importance) {
      return Object.entries(analysisData.feature_importance)
        .map(([feature, importance]) => ({ feature, importance: Number(importance) }))
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 10); // Top 10 features
    }
    return []; // Return empty array instead of mock data
  }, [analysisData]);

  // For backward compatibility with dual charts (A and B)
  const trainScatterDataA = trainScatterData;
  const trainScatterDataB = trainScatterData;
  const validationScatterDataA = validationScatterData;
  const validationScatterDataB = validationScatterData;
  const featureImportanceA = featureImportanceData;
  const featureImportanceB = featureImportanceData;
  
  // 実際のヒートマップデータを使用
  const heatmapDataA = { data: heatmapData, features };
  const heatmapDataB = { data: heatmapData, features };

  // Metrics from analyze-whole endpoint  
  // Get metrics from training_metrics with Japanese keys
  const getMetricsFromTrainingData = () => {
    if (!analysisData?.training_metrics) {
      return { mse_train: 0, r2_train: 0, mse_validate: 0, r2_validate: 0 };
    }
    
    // Try to get metrics from the first available data type
    const trainingMetrics = analysisData.training_metrics;
    const firstKey = Object.keys(trainingMetrics)[0];
    
    if (firstKey && trainingMetrics[firstKey]) {
      return trainingMetrics[firstKey];
    }
    
    return { mse_train: 0, r2_train: 0, mse_validate: 0, r2_validate: 0 };
  };
  
  const trainMetrics = getMetricsFromTrainingData();
  
  const trainRSquaredA = trainMetrics.r2_train;
  const trainRSquaredB = trainMetrics.r2_train;
  const validationRSquaredA = trainMetrics.r2_validate;
  const validationRSquaredB = trainMetrics.r2_validate;
  
  const trainMSEA = trainMetrics.mse_train;
  const trainMSEB = trainMetrics.mse_train;
  const validationMSEA = trainMetrics.mse_validate;
  const validationMSEB = trainMetrics.mse_validate;

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