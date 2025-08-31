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

  // Responsive cell size to fit everything in container
  const maxFeatures = Math.max(features.length, 1);
  const containerWidth = 580; // Increased available width in the card container
  const labelSpace = 140; // Increased space reserved for labels
  const availableWidth = containerWidth - labelSpace;
  const calculatedCellSize = availableWidth / maxFeatures;
  const cellSize = Math.max(6, Math.min(calculatedCellSize, 20)); // Min 6px, Max 20px per cell
  const fontSize = Math.max(6, cellSize * 0.45); // Slightly larger proportional font size

  return (
    <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
      <CardHeader>
        <CardTitle className="text-xl font-semibold text-gray-800 text-center">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center justify-center w-full">
          {/* Main heatmap */}
          <div className="relative w-full max-w-xl ml-4">
            <svg 
              width="100%"
              height={features.length * cellSize + 140}
              className="w-full"
              viewBox={`0 0 ${containerWidth} ${features.length * cellSize + 140}`}
              preserveAspectRatio="xMidYMid meet"
            >
              {/* Y-axis labels */}
              {features.map((feature, i) => (
                <text
                  key={`y-${i}`}
                  x={labelSpace - 10}
                  y={i * cellSize + cellSize / 2 + 60}
                  textAnchor="end"
                  dominantBaseline="middle"
                  fontSize={fontSize}
                  fill="#6B7280"
                  className="select-none"
                >
                  {feature.length > 18 ? `${feature.substring(0, 15)}...` : feature}
                </text>
              ))}

              {/* X-axis labels */}
              {features.map((feature, i) => (
                <text
                  key={`x-${i}`}
                  x={i * cellSize + cellSize / 2 + labelSpace}
                  y={50}
                  textAnchor="start"
                  dominantBaseline="middle"
                  fontSize={fontSize}
                  fill="#6B7280"
                  transform={`rotate(-45, ${i * cellSize + cellSize / 2 + labelSpace}, 50)`}
                  className="select-none"
                >
                  {feature.length > 18 ? `${feature.substring(0, 15)}...` : feature}
                </text>
              ))}

              {/* Heatmap cells */}
              {features.map((yFeature, i) => 
                features.map((xFeature, j) => {
                  const value = correlationMatrix[yFeature]?.[xFeature] || 0;
                  return (
                    <rect
                      key={`${i}-${j}`}
                      x={j * cellSize + labelSpace}
                      y={i * cellSize + 60}
                      width={cellSize}
                      height={cellSize}
                      fill={getColor(value)}
                      stroke="#fff"
                      strokeWidth={0.5}
                      className="hover:stroke-gray-600 hover:stroke-1 transition-all duration-200"
                    >
                      <title>{`${yFeature} vs ${xFeature}: ${value.toFixed(3)}`}</title>
                    </rect>
                  );
                })
              )}
            </svg>
          </div>

          {/* Horizontal Color bar */}
          <div className="mt-4 flex flex-col items-center w-full max-w-xs">
            <div className="relative w-full h-4 rounded shadow-sm" style={{
              background: `linear-gradient(to right, 
                rgb(0, 0, 255) 0%, 
                rgb(255, 255, 255) 50%, 
                rgb(255, 0, 0) 100%)`
            }}>
              {/* Color bar ticks */}
              {[-1, -0.5, 0, 0.5, 1].map((value, index) => (
                <div key={index} className="absolute flex flex-col items-center" style={{ left: `${index * 25}%`, transform: 'translateX(-50%)' }}>
                  <div className="w-0.5 h-3 bg-black mt-4"></div>
                  <span className="text-xs text-gray-600 mt-0.5 font-medium">{value.toFixed(1)}</span>
                </div>
              ))}
            </div>
            <div className="text-xs text-gray-600 mt-6 font-semibold">
              相関係数
            </div>
          </div>
        </div>
        
        <p className="text-xs text-gray-500 text-center mt-3 leading-relaxed">
          特徴量間の相関関係を表示。青は負の相関、赤は正の相関を示します。
        </p>
      </CardContent>
    </Card>
  );
} 