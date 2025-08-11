'use client';

import { useState, useCallback } from 'react';

interface HeatmapData {
  x: string;
  y: string;
  value: number;
}

interface DatasetResponse {
  settlement_data: Record<string, any>[];
  convergence_data: Record<string, any>[];
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export function useHeatmap() {
  const [heatmapData, setHeatmapData] = useState<HeatmapData[]>([]);
  const [features, setFeatures] = useState<string[]>([]);
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const calculateCorrelation = (data: Record<string, any>[], columns: string[]) => {
    if (data.length === 0 || columns.length === 0) return { correlationMatrix: {}, heatmapData: [] };

    // 数値データのみを抽出
    const numericData = data.map(row => {
      const numericRow: Record<string, number> = {};
      columns.forEach(col => {
        const value = row[col];
        if (typeof value === 'number' && !isNaN(value)) {
          numericRow[col] = value;
        }
      });
      return numericRow;
    }).filter(row => Object.keys(row).length > 0);

    if (numericData.length === 0) return { correlationMatrix: {}, heatmapData: [] };

    // 有効な列のみを使用
    const validColumns = columns.filter(col => 
      numericData.some(row => row[col] !== undefined)
    );

    // 相関行列を計算
    const correlationMatrix: Record<string, Record<string, number>> = {};
    const heatmapData: HeatmapData[] = [];

    validColumns.forEach(col1 => {
      correlationMatrix[col1] = {};
      validColumns.forEach(col2 => {
        const values1 = numericData.map(row => row[col1]).filter(v => v !== undefined);
        const values2 = numericData.map(row => row[col2]).filter(v => v !== undefined);
        
        if (values1.length === 0 || values2.length === 0) {
          correlationMatrix[col1][col2] = 0;
        } else if (col1 === col2) {
          correlationMatrix[col1][col2] = 1;
        } else {
          // ピアソン相関係数を計算
          const mean1 = values1.reduce((a, b) => a + b, 0) / values1.length;
          const mean2 = values2.reduce((a, b) => a + b, 0) / values2.length;
          
          let numerator = 0;
          let denominator1 = 0;
          let denominator2 = 0;
          
          for (let i = 0; i < Math.min(values1.length, values2.length); i++) {
            const diff1 = values1[i] - mean1;
            const diff2 = values2[i] - mean2;
            numerator += diff1 * diff2;
            denominator1 += diff1 * diff1;
            denominator2 += diff2 * diff2;
          }
          
          const denominator = Math.sqrt(denominator1 * denominator2);
          correlationMatrix[col1][col2] = denominator === 0 ? 0 : numerator / denominator;
        }
        
        heatmapData.push({
          x: col2,
          y: col1,
          value: correlationMatrix[col1][col2]
        });
      });
    });

    return { correlationMatrix, heatmapData, validColumns };
  };

  const fetchHeatmapData = useCallback(async (folderName: string, maxDistance: number = 100) => {
    setLoading(true);
    setError(null);

    try {
      // API呼び出し
      const response = await fetch(`${API_BASE_URL}/measurements/make-dataset`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          folder_name: folderName,
          max_distance_from_face: maxDistance,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: DatasetResponse = await response.json();
      
      // settlement_dataとconvergence_dataを結合
      const combinedData = [...result.settlement_data, ...result.convergence_data];
      
      if (combinedData.length === 0) {
        throw new Error('No data available');
      }

      // 利用可能な列を取得
      const allColumns = Object.keys(combinedData[0] || {});
      const numericColumns = allColumns.filter(col => {
        return combinedData.some(row => typeof row[col] === 'number' && !isNaN(row[col]));
      });

      setAvailableColumns(allColumns);
      
      // 最大20列に制限
      const featuresToUse = numericColumns.slice(0, 20);
      
      if (featuresToUse.length < 2) {
        throw new Error('Not enough numeric columns for correlation analysis');
      }

      const { heatmapData, validColumns } = calculateCorrelation(combinedData, featuresToUse);
      
      setHeatmapData(heatmapData);
      setFeatures(validColumns || featuresToUse);

      /* 一時的にモックデータを使用（バックアップ）
      const mockFeatures = [
        'TD', 'Distance_from_face', 'Excavation_advance', 'Ground_condition',
        'Support_type', 'Overburden', 'Groundwater', 'Rock_strength',
        'Tunnel_diameter', 'Depth', 'Geological_formation', 'Weather_condition'
      ];

      const mockHeatmapData: HeatmapData[] = [];
      for (let i = 0; i < mockFeatures.length; i++) {
        for (let j = 0; j < mockFeatures.length; j++) {
          let correlation;
          if (i === j) {
            correlation = 1.0;
          } else {
            const baseCorrelation = (Math.random() - 0.5) * 2;
            if (Math.abs(i - j) === 1) {
              correlation = baseCorrelation * 0.7;
            } else if (Math.abs(i - j) <= 3) {
              correlation = baseCorrelation * 0.5;
            } else {
              correlation = baseCorrelation * 0.3;
            }
            correlation = Math.max(-0.95, Math.min(0.95, correlation));
          }
          
          mockHeatmapData.push({
            x: mockFeatures[j],
            y: mockFeatures[i],
            value: correlation
          });
        }
      }

      setHeatmapData(mockHeatmapData);
      setFeatures(mockFeatures);
      setAvailableColumns(mockFeatures);
      */
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      console.error('Error fetching heatmap data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const generateHeatmapImage = useCallback(async (
    folderName: string,
    xColumns: string[],
    yColumn: string,
    correlationMethod: string = 'pearson'
  ) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/charts/draw-heatmap`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          folder_name: folderName,
          x_columns: xColumns,
          y_column: yColumn,
          correlation_method: correlationMethod,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        return result.file_path;
      } else {
        throw new Error(result.message);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      console.error('Error generating heatmap image:', err);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    heatmapData,
    features,
    availableColumns,
    loading,
    error,
    fetchHeatmapData,
    generateHeatmapImage,
  };
} 