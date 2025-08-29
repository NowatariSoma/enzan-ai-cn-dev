'use client';

import { useState, useMemo, useEffect, useCallback } from 'react';
import { useSearchParams } from 'next/navigation';
import { useHeatmap } from './useHeatmap';

interface ProcessEachResult {
  model_name: string;
  data_type: string;
  metrics: {
    mse_train: number;
    r2_train: number;
    mse_validate: number;
    r2_validate: number;
  };
  train_predictions: number[];
  validate_predictions: number[];
  train_actual: number[];
  validate_actual: number[];
  feature_importance?: Record<string, number>;
  scatter_train?: string;
  scatter_validate?: string;
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
  const [processEachData, setProcessEachData] = useState<ProcessEachResult | null>(null);

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

  // Transform process-each data to scatter plot format
  const transformToScatterData = (actual: number[], predicted: number[]) => {
    if (!actual || !predicted) return [];
    return actual.map((actualValue, index) => ({
      actual: actualValue,
      predicted: predicted[index] || 0,
    }));
  };


  const chartData = useMemo(() => generateChartData(), []);
  
  // Use actual data from process-each endpoint if available, otherwise use mock data
  const trainScatterData = useMemo(() => {
    if (processEachData) {
      return transformToScatterData(
        processEachData.train_actual,
        processEachData.train_predictions
      );
    }
    return [];
  }, [processEachData]);

  const validationScatterData = useMemo(() => {
    if (processEachData) {
      return transformToScatterData(
        processEachData.validate_actual,
        processEachData.validate_predictions
      );
    }
    return [];
  }, [processEachData]);

  const featureImportanceData = useMemo(() => {
    if (processEachData?.feature_importance) {
      return Object.entries(processEachData.feature_importance)
        .map(([feature, importance]) => ({ feature, importance }))
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 10); // Top 10 features
    }
    return []; // Return empty array instead of mock data
  }, [processEachData]);

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

  // Metrics from process-each endpoint
  const trainMetrics = processEachData?.metrics || {
    mse_train: 0,
    r2_train: 0,
    mse_validate: 0,
    r2_validate: 0,
  };
  
  const trainRSquaredA = trainMetrics.r2_train;
  const trainRSquaredB = trainMetrics.r2_train;
  const validationRSquaredA = trainMetrics.r2_validate;
  const validationRSquaredB = trainMetrics.r2_validate;
  
  const trainMSEA = trainMetrics.mse_train;
  const trainMSEB = trainMetrics.mse_train;
  const validationMSEA = trainMetrics.mse_validate;
  const validationMSEB = trainMetrics.mse_validate;

  // Fetch data from process-each endpoint
  const fetchProcessEachData = useCallback(async () => {
    setIsAnalyzing(true);
    console.log('Fetching process-each data with params:', {
      model_name: model,
      folder_name: folderName,
      max_distance_from_face: maxDistance,
      data_type: dataType,
      td: predictionTD,
      predict_final: true,
    });

    try {
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
      
      // AbortControllerを使用してタイムアウトを5分に設定
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000); // 5分
      
      const response = await fetch(`${API_BASE_URL}/models/process-each`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_name: model,
          folder_name: folderName,
          max_distance_from_face: maxDistance,
          data_type: dataType,
          td: predictionTD,
          predict_final: true,
        }),
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

      console.log('Response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('API Error:', errorText);
        throw new Error(`Failed to fetch process-each data: ${response.status} ${errorText}`);
      }

      const data: ProcessEachResult = await response.json();
      console.log('Received data:', {
        model_name: data.model_name,
        data_type: data.data_type,
        metrics: data.metrics,
        train_count: data.train_predictions?.length || 0,
        validate_count: data.validate_predictions?.length || 0,
        feature_importance_keys: Object.keys(data.feature_importance || {}),
      });

      setProcessEachData(data);
      
      // Also fetch heatmap data
      fetchHeatmapData(folderName, maxDistance);
    } catch (error) {
      console.error('Error fetching process-each data:', error);
      // Set error state or show user-friendly error
      alert(`API接続エラー: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsAnalyzing(false);
    }
  }, [model, folderName, maxDistance, dataType, predictionTD, fetchHeatmapData]);

  const handleAnalyze = () => {
    fetchProcessEachData();
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
    processEachData,
  };
}