"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface DataSeries {
  name: string;
  data: any[];
  color: string;
  strokeDasharray?: string;
  columns: string[]; // Which columns to display from this dataset
}

interface MultiLineChartProps {
  dataSeries: DataSeries[];
  title: string;
  xAxisLabel: string;
  yAxisLabel: string;
  xAxisKey?: string;
}

export function MultiLineChart({
  dataSeries,
  title,
  xAxisLabel,
  yAxisLabel,
  xAxisKey = "distanceFromFace"
}: MultiLineChartProps) {
  console.log('🔍 MultiLineChart - dataSeries:', dataSeries);
  
  // Create multiple separate LineCharts overlaid on each other
  const colors = ["#3B82F6", "#F59E0B", "#10B981", "#8B5CF6"];
  let colorIndex = 0;

  // Get overall domain for consistent scaling
  const allXValues = new Set<number>();
  const allYValues: number[] = [];
  
  dataSeries.forEach((series) => {
    series.data.forEach(point => {
      if (point[xAxisKey] !== undefined) {
        allXValues.add(point[xAxisKey]);
        series.columns.forEach(column => {
          if (point[column] !== undefined && point[column] !== null) {
            allYValues.push(point[column]);
          }
        });
      }
    });
  });

  const xDomain = [Math.min(...Array.from(allXValues)), Math.max(...Array.from(allXValues))];
  const yDomain = [Math.min(...allYValues) * 1.1, Math.max(...allYValues) * 1.1];

  console.log('🔍 MultiLineChart - domains:', { xDomain, yDomain });

  return (
    <div className="w-full bg-white rounded-lg border p-4 shadow-sm">
      <h3 className="text-lg font-semibold text-gray-800 mb-4 text-center">
        {title}
      </h3>
      <div className="relative w-full h-80">
        {dataSeries.map((series, seriesIndex) =>
          series.columns.map((column, columnIndex) => {
            const lineData = series.data.map(point => ({
              [xAxisKey]: point[xAxisKey],
              value: point[column]
            })).filter(point => 
              point.value !== undefined && 
              point.value !== null &&
              point[xAxisKey] !== undefined
            );

            const color = series.color || colors[colorIndex % colors.length];
            const isFirstChart = seriesIndex === 0 && columnIndex === 0;
            
            console.log(`🔍 Rendering line for ${column} (${series.name}):`, lineData);
            
            const result = (
              <div 
                key={`${series.name}_${column}`}
                className={`absolute inset-0 ${isFirstChart ? '' : 'pointer-events-none'}`}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={lineData}
                    margin={{
                      top: 10,
                      right: 10,
                      left: 40,
                      bottom: 40,
                    }}
                  >
                    {isFirstChart && (
                      <>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis
                          dataKey={xAxisKey}
                          type="number"
                          scale="linear"
                          domain={xDomain}
                          tick={{ fontSize: 12 }}
                          axisLine={{ stroke: "#d1d5db" }}
                          tickLine={{ stroke: "#d1d5db" }}
                          label={{ value: xAxisLabel, position: "insideBottom", offset: -5 }}
                        />
                        <YAxis
                          domain={yDomain}
                          tick={{ fontSize: 12 }}
                          axisLine={{ stroke: "#d1d5db" }}
                          tickLine={{ stroke: "#d1d5db" }}
                          label={{ value: yAxisLabel, angle: -90, position: "insideLeft" }}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "white",
                            border: "1px solid #d1d5db",
                            borderRadius: "8px",
                            boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                          }}
                        />
                      </>
                    )}
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke={color}
                      strokeWidth={2}
                      dot={{ r: 3, fill: color }}
                      strokeDasharray={series.strokeDasharray}
                      name={`${column} (${series.name})`}
                      connectNulls={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            );
            
            colorIndex++;
            return result;
          })
        )}
        
        {/* Legend */}
        <div className="absolute top-2 right-4 bg-white bg-opacity-80 p-2 rounded border text-sm">
          {dataSeries.map((series, seriesIndex) =>
            series.columns.map((column, columnIndex) => {
              const color = series.color || colors[(seriesIndex * series.columns.length + columnIndex) % colors.length];
              return (
                <div key={`legend_${series.name}_${column}`} className="flex items-center gap-2 mb-1">
                  <div 
                    className="w-4 h-0.5" 
                    style={{ 
                      backgroundColor: color,
                      borderStyle: series.strokeDasharray ? 'dashed' : 'solid',
                      borderWidth: series.strokeDasharray ? '1px 0' : '0'
                    }}
                  ></div>
                  <span>{column} ({series.name})</span>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}