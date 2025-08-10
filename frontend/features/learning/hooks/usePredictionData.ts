import { useMemo } from "react";

export function usePredictionData(excavationAdvance: number, distanceFromFace: number) {
  const predictionData = useMemo(() => {
    const data = [];
    for (let i = 0; i <= 6; i++) {
      const days = i === 0 ? 0 : Math.round(i * 5.2125);
      data.push({
        step: i,
        days: days,
        prediction1: (Math.random() * 2 - 1).toFixed(3),
        prediction2: (Math.random() * 0.5 - 0.25).toFixed(3),
        prediction3: (Math.random() * 3 - 1.5).toFixed(3),
      });
    }
    return data;
  }, [excavationAdvance, distanceFromFace]);

  return { predictionData };
}