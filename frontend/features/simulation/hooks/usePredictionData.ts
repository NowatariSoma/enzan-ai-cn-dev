import { useMemo } from "react";

interface PredictionDataPoint {
  step: number;
  days: number;
  distanceFromFace: number;
  [key: string]: number; // For dynamic displacement columns from API
}

export function usePredictionData(
  excavationAdvance: number, 
  distanceFromFace: number, 
  apiData?: Array<{切羽からの距離: number; [key: string]: number}>
) {
  const predictionData = useMemo(() => {
    if (apiData && apiData.length > 0) {
      // Use real API data when available
      return apiData.map((point, index) => {
        const days = index === 0 ? 0 : Math.round((point.切羽からの距離 - distanceFromFace) / excavationAdvance);
        const result: PredictionDataPoint = {
          step: index,
          days: Math.max(0, days),
          distanceFromFace: point.切羽からの距離,
        };
        
        // Add all displacement columns from API data
        Object.keys(point).forEach(key => {
          if (key !== '切羽からの距離') {
            result[key] = point[key];
          }
        });
        
        return result;
      }).slice(0, 20); // Limit to first 20 points for table display
    }

    // Return empty array when no API data is available
    return [];
  }, [excavationAdvance, distanceFromFace, apiData]);

  return { predictionData };
}