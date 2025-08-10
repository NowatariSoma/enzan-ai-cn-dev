'use client';

import { useState, useMemo } from 'react';

export function useSimulation() {
  const [cycleNumber, setCycleNumber] = useState("measurements_A_00004.csv");
  const [distanceFromFace, setDistanceFromFace] = useState(1.0);
  const [excavationAdvance, setExcavationAdvance] = useState(5.0);
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

  const chartData = useMemo(() => generateChartData(), []);

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
    cycleNumber,
    setCycleNumber,
    distanceFromFace,
    setDistanceFromFace,
    excavationAdvance,
    setExcavationAdvance,
    isAnalyzing,
    setIsAnalyzing,
    chartData,
    handleAnalyze,
    chartLines
  };
}