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
  return (
    <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
      <CardHeader>
        <CardTitle className="text-xl font-semibold text-gray-800 text-center">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
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
              fontSize={12}
            />
            <YAxis
              stroke="#6B7280"
              label={{ 
                value: 'Feature Importance', 
                angle: -90, 
                position: 'insideLeft',
                style: { textAnchor: 'middle' }
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
      </CardContent>
    </Card>
  );
} 