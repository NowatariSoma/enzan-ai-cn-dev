'use client';

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/layout/card";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Line, ReferenceLine } from 'recharts';

interface ScatterPlotSectionProps {
  title: string;
  data: Array<{
    actual: number;
    predicted: number;
  }>;
  rSquared: number;
  xLabel: string;
  yLabel: string;
}

export function ScatterPlotSection({ title, data, rSquared, xLabel, yLabel }: ScatterPlotSectionProps) {
  // Calculate min and max values for reference line
  const allValues = data.flatMap(d => [d.actual, d.predicted]);
  const minVal = Math.min(...allValues);
  const maxVal = Math.max(...allValues);

  // Format number to 2 significant digits
  const formatToTwoDigits = (value: number) => {
    return parseFloat(value.toPrecision(2));
  };

  // Format number to 2 significant digits for display
  const formatToTwoDigitsString = (value: any) => {
    return parseFloat(Number(value).toPrecision(2)).toString();
  };

  return (
    <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
      <CardHeader>
        <CardTitle className="text-xl font-semibold text-gray-800 text-center">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart
            margin={{
              top: 10,
              right: 10,
              bottom: 15,
              left: 0,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
            <XAxis
              type="number"
              dataKey="actual"
              name={xLabel}
              domain={[minVal, maxVal]}
              stroke="#6B7280"
              label={{ 
                value: xLabel, 
                position: 'insideBottom', 
                offset: -5
              }}
              tickFormatter={formatToTwoDigitsString}
            />
            <YAxis
              type="number"
              dataKey="predicted"
              name={yLabel}
              domain={[minVal, maxVal]}
              stroke="#6B7280"
              label={{ 
                value: yLabel, 
                angle: -90, 
                position: 'insideLeft',
                style: { textAnchor: 'middle' },
                offset: 10
              }}
              tickFormatter={formatToTwoDigitsString}
            />
            <Tooltip
              contentStyle={{ 
                backgroundColor: 'white', 
                border: '1px solid #E5E7EB',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
              }}
              formatter={(value, name) => [formatToTwoDigits(Number(value)), name]}
              labelFormatter={() => ''}
            />
            
            {/* Reference line (y = x) */}
            <ReferenceLine
              segment={[
                { x: minVal, y: minVal },
                { x: maxVal, y: maxVal }
              ]}
              stroke="#ff4444"
              strokeWidth={2}
            />
            
            {/* Scatter points */}
            <Scatter
              data={data}
              fill="#3B82F6"
              fillOpacity={0.6}
              stroke="#3B82F6"
              strokeWidth={1}
              r={3}
            />
          </ScatterChart>
        </ResponsiveContainer>
        
        {/* R² value display */}
        <p className="text-sm text-gray-500 text-center mt-2">
          R² = {formatToTwoDigits(rSquared)}
        </p>
      </CardContent>
    </Card>
  );
} 