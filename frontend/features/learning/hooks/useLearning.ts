'use client';

import { useState, useMemo } from 'react';

export function useLearning() {
  const [folder, setFolder] = useState("01-hokkaido-akan");
  const [model, setModel] = useState("Random Forest");
  const [predictionTD, setPredictionTD] = useState(500);
  const [maxDistance, setMaxDistance] = useState(100);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

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
    const numPoints = 150; // Slightly fewer points for validation
    
    for (let i = 0; i < numPoints; i++) {
      const actual = (Math.random() - 0.5) * 10; // Random actual value between -5 and 5
      const noise = (Math.random() - 0.5) * 2.5; // Slightly more noise for validation
      const predicted = actual + noise; // Predicted value with some error
      
      data.push({
        actual,
        predicted,
      });
    }
    return data;
  };

  // Generate Feature Importance data
  const generateFeatureImportanceData = () => {
    const features = [
      'TD', 'Distance_from_face', 'Excavation_advance', 'Ground_condition',
      'Support_type', 'Overburden', 'Groundwater', 'Rock_strength',
      'Tunnel_diameter', 'Depth', 'Geological_formation', 'Weather_condition',
      'Equipment_type', 'Advance_rate', 'Face_stability', 'Convergence_rate',
      'Stress_level', 'Deformation_history', 'Support_pressure', 'Time_factor'
    ];

    const data = features.map(feature => ({
      feature,
      importance: Math.random() * 0.15 // Random importance between 0 and 0.15
    }));

    // Sort by importance (descending) and ensure first few have higher values
    data.sort((a, b) => b.importance - a.importance);
    
    // Adjust top features to have more realistic distribution
    if (data.length > 0) data[0].importance = 0.12 + Math.random() * 0.03;
    if (data.length > 1) data[1].importance = 0.08 + Math.random() * 0.03;
    if (data.length > 2) data[2].importance = 0.05 + Math.random() * 0.03;
    if (data.length > 3) data[3].importance = 0.03 + Math.random() * 0.02;

    return data;
  };

  // Generate heatmap data for correlation matrix
  const generateHeatmapData = () => {
    const features = [
      'TD', 'Distance_from_face', 'Excavation_advance', 'Ground_condition',
      'Support_type', 'Overburden', 'Groundwater', 'Rock_strength',
      'Tunnel_diameter', 'Depth', 'Geological_formation', 'Weather_condition',
      'Equipment_type', 'Advance_rate', 'Face_stability', 'Convergence_rate',
      'Stress_level', 'Deformation_history', 'Support_pressure', 'Time_factor'
    ];

    const data = [];
    
    for (let i = 0; i < features.length; i++) {
      for (let j = 0; j < features.length; j++) {
        let correlation;
        
        if (i === j) {
          // Diagonal elements (perfect correlation)
          correlation = 1.0;
        } else {
          // Generate realistic correlation values
          // Some features should have higher correlations
          const baseCorrelation = Math.random() * 2 - 1; // -1 to 1
          
          // Add some structure to make it more realistic
          if (Math.abs(i - j) === 1) {
            // Adjacent features might have higher correlation
            correlation = baseCorrelation * 0.7;
          } else if (Math.abs(i - j) <= 3) {
            // Nearby features might have moderate correlation
            correlation = baseCorrelation * 0.5;
          } else {
            // Distant features have lower correlation
            correlation = baseCorrelation * 0.3;
          }
          
          // Ensure symmetry
          correlation = Math.max(-0.95, Math.min(0.95, correlation));
        }
        
        data.push({
          x: features[j],
          y: features[i],
          value: correlation
        });
      }
    }
    
    return { data, features };
  };

  const chartData = useMemo(() => generateChartData(), []);
  const trainScatterDataA = useMemo(() => generateTrainScatterData(), []);
  const trainScatterDataB = useMemo(() => generateTrainScatterData(), []);
  const validationScatterDataA = useMemo(() => generateValidationScatterData(), []);
  const validationScatterDataB = useMemo(() => generateValidationScatterData(), []);
  const featureImportanceA = useMemo(() => generateFeatureImportanceData(), []);
  const featureImportanceB = useMemo(() => generateFeatureImportanceData(), []);
  const heatmapDataA = useMemo(() => generateHeatmapData(), []);
  const heatmapDataB = useMemo(() => generateHeatmapData(), []);

  // Calculate R² values (mock calculation)
  const calculateRSquared = (data: Array<{actual: number; predicted: number}>) => {
    const actualMean = data.reduce((sum, d) => sum + d.actual, 0) / data.length;
    const totalSumSquares = data.reduce((sum, d) => sum + Math.pow(d.actual - actualMean, 2), 0);
    const residualSumSquares = data.reduce((sum, d) => sum + Math.pow(d.actual - d.predicted, 2), 0);
    
    return Math.max(0, 1 - (residualSumSquares / totalSumSquares));
  };

  const trainRSquaredA = useMemo(() => calculateRSquared(trainScatterDataA), [trainScatterDataA]);
  const trainRSquaredB = useMemo(() => calculateRSquared(trainScatterDataB), [trainScatterDataB]);
  const validationRSquaredA = useMemo(() => calculateRSquared(validationScatterDataA), [validationScatterDataA]);
  const validationRSquaredB = useMemo(() => calculateRSquared(validationScatterDataB), [validationScatterDataB]);

  const handleAnalyze = () => {
    setIsAnalyzing(true);
    setTimeout(() => {
      setIsAnalyzing(false);
    }, 1500);
  };

  const chartLines = [
    { dataKey: "変位量A", stroke: "#3B82F6", name: "変位量A" },
    { dataKey: "変位量B", stroke: "#10B981", name: "変位量B" },
    { dataKey: "変位量C", stroke: "#F59E0B", name: "変位量C" },
    { dataKey: "変位量A_prediction", stroke: "#8B5CF6", name: "変位量A_prediction", strokeDasharray: "5 5" },
    { dataKey: "変位量B_prediction", stroke: "#06B6D4", name: "変位量B_prediction", strokeDasharray: "5 5" },
    { dataKey: "変位量C_prediction", stroke: "#EF4444", name: "変位量C_prediction", strokeDasharray: "5 5" },
  ];

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
    setIsAnalyzing,
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
    chartLines
  };
}