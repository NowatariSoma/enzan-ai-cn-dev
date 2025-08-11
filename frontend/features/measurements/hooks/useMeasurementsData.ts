'use client';

import { useState, useEffect } from 'react';
import { measurementsAPI } from '@/lib/api/measurements';
import type {
  TimeSeriesDataPoint,
  DistributionDataPoint,
  TunnelScatterPoint,
  DistanceDataResponse
} from '@/lib/api/measurements';

export function useMeasurementsData() {
  const [displacementData, setDisplacementData] = useState<TimeSeriesDataPoint[]>([]);
  const [settlementData, setSettlementData] = useState<TimeSeriesDataPoint[]>([]);
  const [displacementDistribution, setDisplacementDistribution] = useState<DistributionDataPoint[]>([]);
  const [settlementDistribution, setSettlementDistribution] = useState<DistributionDataPoint[]>([]);
  const [scatterData, setScatterData] = useState<TunnelScatterPoint[]>([]);
  const [distanceData, setDistanceData] = useState<DistanceDataResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // distance-data APIのデータを既存のグラフ形式に変換
  const transformDistanceDataToTimeSeries = (
    distanceData: DistanceDataResponse,
    dataType: 'settlements' | 'convergences'
  ): TimeSeriesDataPoint[] => {
    const result: TimeSeriesDataPoint[] = [];
    const distances = ['3m', '5m', '10m', '20m', '50m', '100m'];
    
    // 全距離のTDデータを統合
    const allTDs = new Set<number>();
    distances.forEach(distance => {
      const data = distanceData.dct_df_td[distance];
      if (data) {
        data.forEach(point => allTDs.add(point.td));
      }
    });
    
    // TD順にソート
    const sortedTDs = Array.from(allTDs).sort((a, b) => a - b);
    
    // 各TDポイントでのデータを作成
    sortedTDs.forEach(td => {
      const point: TimeSeriesDataPoint = {
        td,
        series3m: 0,
        series5m: 0,
        series10m: 0,
        series20m: 0,
        series50m: 0,
        series100m: 0,
      };
      
      distances.forEach(distance => {
        const data = distanceData.dct_df_td[distance];
        if (data) {
          const tdPoint = data.find(p => p.td === td);
          if (tdPoint) {
            const values = dataType === 'settlements' ? tdPoint.settlements : tdPoint.convergences;
            // 複数の値がある場合は平均を取る
            const avg = values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;
            const seriesKey = `series${distance}` as keyof TimeSeriesDataPoint;
            point[seriesKey] = avg;
          }
        }
      });
      
      result.push(point);
    });
    
    return result;
  };

  // distance-data APIのデータをDistribution形式に変換
  const transformDistanceDataToDistribution = (
    distanceData: DistanceDataResponse,
    dataType: 'settlements' | 'convergences'
  ): DistributionDataPoint[] => {
    const result: DistributionDataPoint[] = [];
    const distances = ['3m', '5m', '10m', '20m', '50m', '100m'];
    
    // 全データから最小値と最大値を計算
    let globalMin = Infinity;
    let globalMax = -Infinity;
    
    distances.forEach(distance => {
      const data = dataType === 'settlements' 
        ? distanceData.settlements[distance]
        : distanceData.convergences[distance];
      
      if (data && Array.isArray(data)) {
        data.forEach(value => {
          if (value < globalMin) globalMin = value;
          if (value > globalMax) globalMax = value;
        });
      }
    });
    
    // データが存在しない場合のデフォルト値
    if (globalMin === Infinity || globalMax === -Infinity) {
      globalMin = dataType === 'settlements' ? -30 : -15;
      globalMax = dataType === 'settlements' ? 0 : 15;
    }
    
    // 最小値と最大値を整数に丸める（Pythonのrange(int(min), int(max)+2)と同じ）
    const minBin = Math.floor(globalMin);
    const maxBin = Math.ceil(globalMax) + 1; // Pythonのrange(..., max+2)と同等
    
    // 0.5mm刻みでビンを生成（より細かい分割）
    const binSize = 0.5;
    for (let bin = minBin; bin < maxBin; bin += binSize) {
      const binStart = bin;
      const binEnd = bin + binSize;
      const range = `${binStart.toFixed(1)}~${binEnd.toFixed(1)}`;
      
      const point: DistributionDataPoint = {
        range,
        series3m: 0,
        series5m: 0,
        series10m: 0,
        series20m: 0,
        series50m: 0,
        series100m: 0,
      };
      
      distances.forEach(distance => {
        const data = dataType === 'settlements' 
          ? distanceData.settlements[distance]
          : distanceData.convergences[distance];
        
        if (data && Array.isArray(data)) {
          let count = 0;
          data.forEach(value => {
            if (value >= binStart && value < binEnd) {
              count++;
            }
          });
          const seriesKey = `series${distance}` as keyof DistributionDataPoint;
          (point as any)[seriesKey] = count;
        }
      });
      
      result.push(point);
    }
    
    return result;
  };

  useEffect(() => {
    const fetchAllData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // distance-data APIからデータ取得
        const distanceRes = await measurementsAPI.getDistanceData('01-hokkaido-akan', 100);
        setDistanceData(distanceRes);
        
        // distance-dataを既存のグラフ形式に変換
        const transformedDisplacement = transformDistanceDataToTimeSeries(distanceRes, 'convergences');
        const transformedSettlement = transformDistanceDataToTimeSeries(distanceRes, 'settlements');
        
        setDisplacementData(transformedDisplacement);
        setSettlementData(transformedSettlement);
        
        // Distribution データも distance-data から生成
        const transformedDisplacementDist = transformDistanceDataToDistribution(distanceRes, 'convergences');
        const transformedSettlementDist = transformDistanceDataToDistribution(distanceRes, 'settlements');
        
        setDisplacementDistribution(transformedDisplacementDist);
        setSettlementDistribution(transformedSettlementDist);
        
        // Scatter データは従来のAPIから取得（モックまたは実装済みの場合）
        try {
          const scatterRes = await measurementsAPI.getTunnelScatter(800);
          setScatterData(scatterRes.data);
        } catch (err) {
          // APIが未実装の場合はダミーデータを設定
          console.warn('Scatter API is not available, using dummy data');
          setScatterData([]);
        }
      } catch (err) {
        console.error('Error fetching measurements data:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    };

    fetchAllData();
  }, []);


  return {
    displacementData,
    settlementData,
    displacementDistribution,
    settlementDistribution,
    scatterData,
    distanceData,
    loading,
    error
  };
}