'use client';

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/layout/card";
import { useMemo } from 'react';

interface HeatmapData {
  x: string;
  y: string;
  value: number;
}

interface HeatmapSectionProps {
  title: string;
  data: HeatmapData[];
  features: string[];
  loading?: boolean;
  error?: string | null;
}

export function HeatmapSection({ title, data, features }: HeatmapSectionProps) {
  // Create a color scale function
  const getColor = (value: number) => {
    // Normalize value from -1 to 1 to 0 to 1
    const normalized = (value + 1) / 2;
    
    if (normalized <= 0.5) {
      // Blue to white
      const intensity = normalized * 2;
      const red = Math.round(255 * intensity);
      const green = Math.round(255 * intensity);
      const blue = 255;
      return `rgb(${red}, ${green}, ${blue})`;
    } else {
      // White to red
      const intensity = (normalized - 0.5) * 2;
      const red = 255;
      const green = Math.round(255 * (1 - intensity));
      const blue = Math.round(255 * (1 - intensity));
      return `rgb(${red}, ${green}, ${blue})`;
    }
  };

  // Create a grid of correlation values
  const correlationMatrix = useMemo(() => {
    const matrix: { [key: string]: { [key: string]: number } } = {};
    
    features.forEach(feature => {
      matrix[feature] = {};
    });

    data.forEach(item => {
      if (!matrix[item.y]) matrix[item.y] = {};
      matrix[item.y][item.x] = item.value;
    });

    return matrix;
  }, [data, features]);

  const cellSize = 18; // Reduced from 25 to 18
  const fontSize = 8; // Reduced font size for labels

  return (
    <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
      <CardHeader>
        <CardTitle className="text-xl font-semibold text-gray-800 text-center">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center justify-center">
          {/* Main heatmap */}
          <div className="relative">
            <svg 
              width={features.length * cellSize + 30} // Reduced margin from 50 to 30 (20px left)
              height={features.length * cellSize + 140} // Increased height from 120 to 140 (20px down)
              className="overflow-visible"
            >
              {/* Y-axis labels */}
              {features.map((feature, i) => (
                <text
                  key={`y-${i}`}
                  x={25} // Adjusted position from 45 to 25 (20px left)
                  y={i * cellSize + cellSize / 2 + 60} // Adjusted position from 40 to 60 (20px down)
                  textAnchor="end"
                  dominantBaseline="middle"
                  fontSize={fontSize}
                  fill="#6B7280"
                  transform={`rotate(-45, 25, ${i * cellSize + cellSize / 2 + 60})`}
                >
                  {feature}
                </text>
              ))}

              {/* X-axis labels */}
              {features.map((feature, i) => (
                <text
                  key={`x-${i}`}
                  x={i * cellSize + cellSize / 2 + 30} // Adjusted position from 50 to 30 (20px left)
                  y={55} // Adjusted position from 35 to 55 (20px down)
                  textAnchor="start"
                  dominantBaseline="middle"
                  fontSize={fontSize}
                  fill="#6B7280"
                  transform={`rotate(-45, ${i * cellSize + cellSize / 2 + 30}, 55)`}
                >
                  {feature}
                </text>
              ))}

              {/* Heatmap cells */}
              {features.map((yFeature, i) => 
                features.map((xFeature, j) => {
                  const value = correlationMatrix[yFeature]?.[xFeature] || 0;
                  return (
                    <rect
                      key={`${i}-${j}`}
                      x={j * cellSize + 30} // Adjusted position from 50 to 30 (20px left)
                      y={i * cellSize + 60} // Adjusted position from 40 to 60 (20px down)
                      width={cellSize}
                      height={cellSize}
                      fill={getColor(value)}
                      stroke="#fff"
                      strokeWidth={0.5}
                    />
                  );
                })
              )}
            </svg>
          </div>

          {/* Horizontal Color bar */}
          <div className="mt-1 flex flex-col items-center">
            <div className="relative w-48 h-5 rounded" style={{
              background: `linear-gradient(to right, 
                rgb(0, 0, 255) 0%, 
                rgb(255, 255, 255) 50%, 
                rgb(255, 0, 0) 100%)`
            }}>
              {/* Color bar ticks */}
              {[-1, -0.5, 0, 0.5, 1].map((value, index) => (
                <div key={index} className="absolute flex flex-col items-center" style={{ left: `${index * 25}%`, transform: 'translateX(-50%)' }}>
                  <div className="w-0.5 h-2 bg-black mt-5"></div>
                  <span className="text-xs text-gray-600 mt-1">{value.toFixed(1)}</span>
                </div>
              ))}
            </div>
            <div className="text-xs text-gray-600 mt-8">
              Correlation
            </div>
          </div>
        </div>
        
        <p className="text-sm text-gray-500 text-center mt-4">
          X列特徴量（座標・変位）間の相関関係
        </p>
      </CardContent>
    </Card>
  );
} 