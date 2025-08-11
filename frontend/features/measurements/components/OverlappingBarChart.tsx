'use client';

import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

interface OverlappingBarChartProps {
  data: any[];
  xKey: string;
  yKeys: string[];
  colors: string[];
  names: string[];
  xLabel?: string;
  yLabel?: string;
}

// カスタムバーシェイプで重なりを実現
const CustomBar = (props: any) => {
  const { fill, x, y, width, height } = props;
  return <rect x={x} y={y} width={width} height={height} fill={fill} opacity={0.5} />;
};

export function OverlappingBarChart({ 
  data, 
  xKey, 
  yKeys, 
  colors, 
  names,
  xLabel = '',
  yLabel = ''
}: OverlappingBarChartProps) {
  // 各系列の最大値を計算してメインのデータとする
  const combinedData = data.map(item => {
    const maxValue = Math.max(...yKeys.map(key => item[key] || 0));
    return {
      ...item,
      maxValue,
      // 各系列の値も保持
      ...yKeys.reduce((acc, key, index) => ({
        ...acc,
        [`_${key}`]: item[key] || 0
      }), {})
    };
  });

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* ベースのグラフ（軸とグリッド） */}
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 30, left: 40, bottom: 80 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
          <XAxis
            dataKey={xKey}
            stroke="#6B7280"
            label={{ value: xLabel, position: 'insideBottom', offset: -5 }}
            angle={-45}
            textAnchor="end"
            height={80}
            interval={Math.floor(data.length / 20)}
          />
          <YAxis
            stroke="#6B7280"
            label={{ value: yLabel, angle: -90, position: 'insideLeft' }}
            domain={['dataMin', 'dataMax']}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #E5E7EB',
              borderRadius: '8px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
            }}
            content={({ active, payload, label }) => {
              if (active && payload && payload.length > 0) {
                const dataPoint = data.find(d => d[xKey] === label);
                return (
                  <div className="p-2 bg-white rounded shadow">
                    <p className="font-semibold">{label}</p>
                    {yKeys.map((k, i) => (
                      <p key={k} style={{ color: colors[i] }}>
                        {names[i]}: {dataPoint?.[k] || 0}
                      </p>
                    ))}
                  </div>
                );
              }
              return null;
            }}
          />
        </BarChart>
      </ResponsiveContainer>
      
      {/* 各系列を重ねて表示 */}
      {yKeys.map((key, index) => (
        <div
          key={key}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            pointerEvents: 'none'
          }}
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 20, right: 30, left: 40, bottom: 80 }}>
              <XAxis 
                dataKey={xKey} 
                hide 
                angle={-45}
                textAnchor="end"
                height={80}
                interval={Math.floor(data.length / 20)}
              />
              <YAxis hide domain={['dataMin', 'dataMax']} />
              <Bar
                dataKey={key}
                fill={colors[index]}
                opacity={0.5}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ))}
      {/* 凡例 */}
      <div className="flex justify-center mt-4 space-x-4" style={{ position: 'relative', zIndex: 10 }}>
        {names.map((name, index) => (
          <div key={name} className="flex items-center">
            <div
              className="w-4 h-4 mr-2"
              style={{ backgroundColor: colors[index], opacity: 0.5 }}
            />
            <span className="text-sm">{name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}