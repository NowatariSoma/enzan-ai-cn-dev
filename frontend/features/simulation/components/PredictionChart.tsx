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
  columns: string[];
}

interface PredictionChartProps {
  data?: any[]; // For backward compatibility
  title: string;
  xAxisLabel: string;
  yAxisLabel: string;
  lines?: {
    dataKey: string;
    stroke: string;
    name: string;
    strokeDasharray?: string;
  }[];
  // New multi-series support
  dataSeries?: DataSeries[];
  xAxisKey?: string;
}

export function PredictionChart({
  data,
  title,
  xAxisLabel,
  yAxisLabel,
  lines,
  dataSeries,
  xAxisKey = "distanceFromFace"
}: PredictionChartProps) {
  
  // If dataSeries is provided, use multi-series rendering
  if (dataSeries) {
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

    const xDomain = [0, 100];
    const yDomain = allYValues.length > 0 ? [Math.min(...allYValues) * 1.1, Math.max(...allYValues) * 1.1] : [0, 1];

    // Create unified dataset with all series data, limited to 0-100 range
    const filteredXValues = Array.from(allXValues).filter(x => x >= 0 && x <= 100);
    const step = 0.5; // Step size for x-axis points
    
    // Include filtered data points plus fill in the 0-100 range
    const xValuesSet = new Set(filteredXValues);
    for (let x = 0; x <= 100; x += step) {
      xValuesSet.add(Number(x.toFixed(1)));
    }
    
    const unifiedData = Array.from(xValuesSet).sort((a, b) => a - b).map(xValue => {
      const dataPoint: any = { [xAxisKey]: xValue };
      
      dataSeries.forEach((series) => {
        let sourcePoint = series.data.find(p => p[xAxisKey] === xValue);
        
        // If no exact match and xValue > max data point, use the last available data point (extrapolation)
        if (!sourcePoint && xValue > Math.max(...series.data.map(p => p[xAxisKey]))) {
          const sortedData = series.data
            .filter(p => p[xAxisKey] !== undefined)
            .sort((a, b) => a[xAxisKey] - b[xAxisKey]);
          sourcePoint = sortedData[sortedData.length - 1]; // Use last data point for extrapolation
        }
        
        if (sourcePoint) {
          series.columns.forEach(column => {
            if (sourcePoint[column] !== undefined && sourcePoint[column] !== null) {
              const key = `${series.name}_${column}`;
              dataPoint[key] = sourcePoint[column];
            }
          });
        }
      });
      
      return dataPoint;
    });

    return (
      <div className="w-full bg-white rounded-lg border p-4 shadow-sm">
        <h3 className="text-lg font-semibold text-gray-800 mb-4 text-center">
          {title}
        </h3>
        <div className="relative">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={unifiedData}
              margin={{
                top: 10,
                right: 10,
                left: 20,
                bottom: 40,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey={xAxisKey}
                type="number"
                scale="linear"
                domain={[0, 100]}
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
              <Legend />
              
              {dataSeries.map((series, seriesIndex) =>
                series.columns.map((column, columnIndex) => {
                  const color = series.color || colors[colorIndex % colors.length];
                  const key = `${series.name}_${column}`;
                  
                  colorIndex++;
                  
                  return (
                    <Line
                      key={key}
                      type="monotone"
                      dataKey={key}
                      stroke={color}
                      strokeWidth={2}
                      dot={false}
                      strokeDasharray={series.strokeDasharray}
                      name={`${column} (${series.name})`}
                      connectNulls={true}
                    />
                  );
                })
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  }

  // Fallback to original single-series rendering
  return (
    <div className="w-full bg-white rounded-lg border p-4 shadow-sm">
      <h3 className="text-lg font-semibold text-gray-800 mb-4 text-center">
        {title}
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart
          data={data}
          margin={{
            top: 10,
            right: 10,
            left: 20,
            bottom: 40,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="distanceFromFace"
            type="number"
            scale="linear"
            domain={[0, 100]}
            tick={{ fontSize: 12 }}
            axisLine={{ stroke: "#d1d5db" }}
            tickLine={{ stroke: "#d1d5db" }}
            label={{ value: xAxisLabel, position: "insideBottom", offset: -5 }}
          />
          <YAxis
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
              boxShadow: "0 4px 6px -1px rgba(0, 0, 1, 0.1)",
            }}
          />
          <Legend />
          {(lines || []).map((line) => (
            <Line
              key={line.dataKey}
              type="monotone"
              dataKey={line.dataKey}
              stroke={line.stroke}
              strokeWidth={2}
              dot={false}
              strokeDasharray={line.strokeDasharray}
              name={line.name}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
} 