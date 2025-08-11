'use client';

import { useState, useMemo, useEffect } from 'react';
import { useHeatmap } from './useHeatmap';

export function useLearning() {
  const [folder, setFolder] = useState("01-hokkaido-akan");
  const [model, setModel] = useState("Random Forest");
  const [predictionTD, setPredictionTD] = useState(500);
  const [maxDistance, setMaxDistance] = useState(100);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // useHeatmapフックを使用
  const { heatmapData, features, fetchHeatmapData } = useHeatmap();

  // フォルダが変更されたときにヒートマップデータを取得
  useEffect(() => {
    fetchHeatmapData(folder, maxDistance);
  }, [folder, maxDistance, fetchHeatmapData]);

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

  // Generate scatter plot data for Actual vs Predicted (Train Data)
  const generateTrainScatterData = () => {
    const data = [];
    const numPoints = 200;
    
    for (let i = 0; i < numPoints; i++) {
      const actual = (Math.random() - 0.5) * 10; // Random actual value between -5 and 5
      const noise = (Math.random() - 0.5) * 2; // Add some noise
      const predicted = actual + noise; // Predicted value with some error
      
      data.push({
        actual,
        predicted,
      });
    }
    return data;
  };

  // Generate scatter plot data for Actual vs Predicted (Validation Data)
  const generateValidationScatterData = () => {
    const data = [];
    const numPoints = 100;
    
    for (let i = 0; i < numPoints; i++) {
      const actual = (Math.random() - 0.5) * 10;
      const noise = (Math.random() - 0.5) * 3; // More noise for validation data
      const predicted = actual + noise;
      
      data.push({
        actual,
        predicted,
      });
    }
    return data;
  };

  // Generate feature importance data
  const generateFeatureImportanceData = () => {
    const features = [
      'TD', 'Distance_from_face', 'Excavation_advance', 'Ground_condition',
      'Support_type', 'Overburden', 'Groundwater', 'Rock_strength',
      'Tunnel_diameter', 'Depth', 'Geological_formation', 'Weather_condition',
      'Equipment_type', 'Advance_rate', 'Face_stability', 'Convergence_rate'
    ];

    const data = features.map(feature => ({
      feature,
      importance: Math.random() * 0.15 + 0.01 // Random importance between 0.01 and 0.16
    }));

    // Sort by importance in descending order
    data.sort((a, b) => b.importance - a.importance);

    return data;
  };

  const chartData = useMemo(() => generateChartData(), []);
  const trainScatterDataA = useMemo(() => generateTrainScatterData(), []);
  const trainScatterDataB = useMemo(() => generateTrainScatterData(), []);
  const validationScatterDataA = useMemo(() => generateValidationScatterData(), []);
  const validationScatterDataB = useMemo(() => generateValidationScatterData(), []);
  const featureImportanceA = useMemo(() => generateFeatureImportanceData(), []);
  const featureImportanceB = useMemo(() => generateFeatureImportanceData(), []);
  
  // 実際のヒートマップデータを使用
  const heatmapDataA = { data: heatmapData, features };
  const heatmapDataB = { data: heatmapData, features };

  // R² scores
  const trainRSquaredA = 0.876;
  const trainRSquaredB = 0.823;
  const validationRSquaredA = 0.792;
  const validationRSquaredB = 0.745;

  const handleAnalyze = () => {
    setIsAnalyzing(true);
    // 分析処理のシミュレーション
    setTimeout(() => {
      setIsAnalyzing(false);
      // 新しいヒートマップデータを取得
      fetchHeatmapData(folder, maxDistance);
    }, 3000);
  };

  return {
    folder,
    setFolder,
    model,
    setModel,
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
    handleAnalyze,
  };
}