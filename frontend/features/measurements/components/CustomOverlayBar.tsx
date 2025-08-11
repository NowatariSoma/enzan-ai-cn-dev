'use client';

import React from 'react';
import { Rectangle } from 'recharts';

// カスタムバーコンポーネント - 完全に重なるように描画
export const CustomOverlayBar = (props: any) => {
  const { fill, x, y, width, height, payload, dataKey } = props;
  
  // 全てのデータキーを取得
  const allDataKeys = ['series3m', 'series5m', 'series10m', 'series20m', 'series50m', 'series100m'];
  const colors = ['#3B82F6', '#F59E0B', '#10B981', '#EF4444', '#8B5CF6', '#6B7280'];
  
  // 現在のバーが最初のシリーズの場合のみ、全シリーズを描画
  if (dataKey === 'series3m') {
    return (
      <g>
        {allDataKeys.map((key, index) => {
          const value = payload[key];
          if (!value || value === 0) return null;
          
          // 高さを計算（yスケールに基づく）
          const barHeight = (value / Math.max(...allDataKeys.map(k => payload[k] || 0))) * height;
          const barY = y + height - barHeight;
          
          return (
            <Rectangle
              key={key}
              x={x}
              y={barY}
              width={width}
              height={barHeight}
              fill={colors[index]}
              fillOpacity={0.5}
            />
          );
        })}
      </g>
    );
  }
  
  // 他のシリーズは描画しない（series3mで全て描画済み）
  return null;
};

// 統合バーチャート用のカスタムバー
export const OverlayBar = (props: any) => {
  const { fill, x, y, width, height, payload } = props;
  
  const allDataKeys = ['series3m', 'series5m', 'series10m', 'series20m', 'series50m', 'series100m'];
  const colors = ['#3B82F6', '#F59E0B', '#10B981', '#EF4444', '#8B5CF6', '#6B7280'];
  
  // 各シリーズのバーを同じ位置に重ねて描画
  return (
    <g>
      {allDataKeys.map((key, index) => {
        const value = payload[key];
        if (!value || value === 0) return null;
        
        return (
          <rect
            key={key}
            x={x}
            y={y}
            width={width}
            height={height}
            fill={colors[index]}
            fillOpacity={0.5}
          />
        );
      })}
    </g>
  );
};