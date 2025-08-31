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
  mse?: number;
  xLabel: string;
  yLabel: string;
}

export function ScatterPlotSection({ title, data, rSquared, mse, xLabel, yLabel }: ScatterPlotSectionProps) {
  // Check if data is empty or invalid
  const hasValidData = data && data.length > 0;
  
  // Calculate min and max values for reference line
  const allValues = hasValidData ? data.flatMap(d => [d.actual, d.predicted]) : [0];
  const minVal = hasValidData ? Math.min(...allValues) : 0;
  const maxVal = hasValidData ? Math.max(...allValues) : 1;

  // Format number to 2 significant digits
  const formatToTwoDigits = (value: number | undefined) => {
    if (value === undefined || value === null || isNaN(value)) return 'N/A';
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
        {hasValidData ? (
          <>
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
            
            {/* Metrics display */}
            <div className="flex justify-center gap-4 mt-2">
              <p className="text-sm text-gray-600">
                MSE: <span className="font-semibold">{mse !== undefined ? formatToTwoDigits(mse) : 'N/A'}</span>
              </p>
              <p className="text-sm text-gray-600">
                R²: <span className="font-semibold">{formatToTwoDigits(rSquared)}</span>
              </p>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center h-64 text-gray-500">
            <div className="mb-4">
              <svg className="w-16 h-16 mx-auto text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <p className="text-lg font-medium mb-2">データがありません</p>
            <p className="text-sm text-center">
              「予測モデル作成を実行」ボタンを<br />
              クリックしてデータを読み込んでください
            </p>
            
            {/* Show current parameters */}
            <div className="mt-4 p-3 bg-gray-50 rounded-lg text-xs">
              <p className="font-medium mb-1">現在の設定:</p>
              <p>MSE: {mse !== undefined ? formatToTwoDigits(mse) : 'N/A'}</p>
              <p>R²: {formatToTwoDigits(rSquared)}</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 