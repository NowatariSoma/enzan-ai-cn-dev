'use client';

import { useState, useEffect } from 'react';
import { measurementsAPI } from '@/lib/api/measurements';
import type {
  TimeSeriesDataPoint,
  DistributionDataPoint,
  TunnelScatterPoint
} from '@/lib/api/measurements';

export function useMeasurementsData() {
  const [displacementData, setDisplacementData] = useState<TimeSeriesDataPoint[]>([]);
  const [settlementData, setSettlementData] = useState<TimeSeriesDataPoint[]>([]);
  const [displacementDistribution, setDisplacementDistribution] = useState<DistributionDataPoint[]>([]);
  const [settlementDistribution, setSettlementDistribution] = useState<DistributionDataPoint[]>([]);
  const [scatterData, setScatterData] = useState<TunnelScatterPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAllData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // Fetch all data in parallel
        const [
          displacementRes,
          settlementRes,
          displacementDistRes,
          settlementDistRes,
          scatterRes
        ] = await Promise.all([
          measurementsAPI.getDisplacementSeries(60), // 60 points for TD up to 1200
          measurementsAPI.getSettlementSeries(60),
          measurementsAPI.getDisplacementDistribution(),
          measurementsAPI.getSettlementDistribution(),
          measurementsAPI.getTunnelScatter(800)
        ]);

        setDisplacementData(displacementRes.data);
        setSettlementData(settlementRes.data);
        setDisplacementDistribution(displacementDistRes.data);
        setSettlementDistribution(settlementDistRes.data);
        setScatterData(scatterRes.data);
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
    loading,
    error
  };
}