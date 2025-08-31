'use client';

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/layout/card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface FeatureImportanceData {
  feature: string;
  importance: number;
}

interface FeatureImportanceSectionProps {
  title: string;
  data: FeatureImportanceData[];
}

export function FeatureImportanceSection({ title, data }: FeatureImportanceSectionProps) {
  const hasValidData = data && data.length > 0;

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
              <BarChart
                data={data}
                margin={{
                  top: 10,
                  right: 10,
                  bottom: 10,
                  left: 0,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis
                  dataKey="feature"
                  stroke="#6B7280"
                  angle={-45}
                  textAnchor="end"
                  height={100}
                  interval={0}
                  fontSize={8}
                />
                <YAxis
                  stroke="#6B7280"
                  fontSize={10}
                  label={{ 
                    value: 'Feature Importance', 
                    angle: -90, 
                    position: 'insideLeft',
                    style: { textAnchor: 'middle', fontSize: '10px' }
                  }}
                />
                <Tooltip
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #E5E7EB',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }}
                  formatter={(value, name) => [
                    typeof value === 'number' ? value.toFixed(3) : value,
                    'Importance'
                  ]}
                />
                <Bar 
                  dataKey="importance" 
                  fill="#3B82F6"
                  radius={[2, 2, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
            
            <p className="text-sm text-gray-500 text-center mt-2">
              特徴量の重要度分析
            </p>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center h-64 text-gray-500">
            <div className="mb-4">
              <svg className="w-16 h-16 mx-auto text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <p className="text-lg font-medium mb-2">データがありません</p>
            <p className="text-sm text-center">
              「予測モデル作成を実行」ボタンを<br />
              クリックして特徴量重要度を計算してください
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 