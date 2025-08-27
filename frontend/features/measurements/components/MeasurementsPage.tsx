'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/layout/card';
import { Skeleton } from '@/components/ui/feedback/skeleton';
import { ErrorAlert } from '@/shared/components/common/ErrorAlert';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, Cell, Rectangle } from 'recharts';
import { useMeasurementsData } from '../hooks/useMeasurementsData';

// カスタムバーシェイプ - 全シリーズを同じ位置に重ねて描画
const CustomOverlayShape = (props: any) => {
  const { x, y, width, height, payload, dataKey } = props;
  
  // dataKeyがdummyBaseの場合のみ全シリーズを描画
  if (dataKey !== 'dummyBase') return <g />;
  
  const allDataKeys = ['series3m', 'series5m', 'series10m', 'series20m', 'series50m', 'series100m'];
  const colors = ['#3B82F6', '#F59E0B', '#10B981', '#EF4444', '#8B5CF6', '#6B7280'];
  
  // heightはdummyBaseの値に対応する高さなので、そのまま使用
  const baseValue = payload.dummyBase || 1;
  
  return (
    <g>
      {allDataKeys.map((key, index) => {
        const value = payload[key];
        if (!value || value === 0) return null;
        
        // dummyBaseを基準に各バーの高さを計算
        const ratio = value / baseValue;
        const barHeight = height * ratio;
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
};

// Color bar component
const ColorBar = () => {
  const colorStops = [
    { value: 15, color: '#8B0000' },
    { value: 10, color: '#FF4500' },
    { value: 5, color: '#FFA500' },
    { value: 0, color: '#FFFF00' },
    { value: -5, color: '#00FF00' },
    { value: -10, color: '#00FFFF' },
    { value: -15, color: '#0080FF' },
    { value: -20, color: '#0000FF' },
    { value: -25, color: '#000080' },
    { value: -30, color: '#000040' }
  ];

  return (
    <div className="flex flex-col items-center ml-4">
      <div className="relative h-64 w-6 rounded" style={{
        background: `linear-gradient(to bottom, 
          #8B0000 0%, 
          #FF4500 11%, 
          #FFA500 22%, 
          #FFFF00 33%, 
          #00FF00 44%, 
          #00FFFF 55%, 
          #0080FF 66%, 
          #0000FF 77%, 
          #000080 88%, 
          #000040 100%)`
      }}>
        {colorStops.map((stop, index) => (
          <div key={index} className="absolute flex items-center" style={{ top: `${((15 - stop.value) / 45) * 100}%` }}>
            <div className="w-2 h-0.5 bg-black ml-6"></div>
            <span className="text-xs ml-1 text-gray-600">{stop.value}</span>
          </div>
        ))}
      </div>
      <div className="text-xs text-gray-600 mt-2 transform -rotate-90 origin-center">
        深度
      </div>
    </div>
  );
};

// Loading skeleton for charts
const ChartSkeleton = () => (
  <div className="w-full h-[300px] flex items-center justify-center">
    <div className="space-y-2 w-full">
      <Skeleton className="h-4 w-3/4" />
      <Skeleton className="h-[250px] w-full" />
      <Skeleton className="h-4 w-1/2 mx-auto" />
    </div>
  </div>
);

// カスタムドット（小さいサイズ）
const CustomDot = (props: any) => {
  const { cx, cy, fill } = props;
  return (
    <circle cx={cx} cy={cy} r={2} fill={fill} />
  );
};

// 値を色に変換するヘルパー関数
const getColorFromValue = (value: number, min: number, max: number): string => {
  // 正規化（0-1の範囲に変換）
  const normalized = max > min ? (value - min) / (max - min) : 0.5;
  
  // JETカラーマップ風の色生成
  let r, g, b;
  if (normalized < 0.25) {
    // 青から水色へ
    r = 0;
    g = Math.floor(normalized * 4 * 255);
    b = 255;
  } else if (normalized < 0.5) {
    // 水色から緑へ
    r = 0;
    g = 255;
    b = Math.floor((0.5 - normalized) * 4 * 255);
  } else if (normalized < 0.75) {
    // 緑から黄色へ
    r = Math.floor((normalized - 0.5) * 4 * 255);
    g = 255;
    b = 0;
  } else {
    // 黄色から赤へ
    r = 255;
    g = Math.floor((1 - normalized) * 4 * 255);
    b = 0;
  }
  
  return `rgb(${r}, ${g}, ${b})`;
};

export function MeasurementsPage() {
  const {
    displacementData,
    settlementData,
    displacementDistribution,
    settlementDistribution,
    scatterData,
    convergenceScatterData,
    settlementScatterData,
    loading,
    error
  } = useMeasurementsData();

  // 変位量散布図データに色を追加
  const convergenceScatterWithColor = convergenceScatterData.map(point => {
    const min = Math.min(...convergenceScatterData.map(p => p.value));
    const max = Math.max(...convergenceScatterData.map(p => p.value));
    return {
      ...point,
      color: getColorFromValue(point.value, min, max)
    };
  });

  // 沈下量散布図データに色を追加
  const settlementScatterWithColor = settlementScatterData.map(point => {
    const min = Math.min(...settlementScatterData.map(p => p.value));
    const max = Math.max(...settlementScatterData.map(p => p.value));
    return {
      ...point,
      color: getColorFromValue(point.value, min, max)
    };
  });

  // デバッグ用ログ
  console.log('Convergence scatter data points:', convergenceScatterData.length);
  console.log('Settlement scatter data points:', settlementScatterData.length);
  if (convergenceScatterData.length > 0) {
    console.log('Sample convergence point:', convergenceScatterWithColor[0]);
  }


  // Show error state
  if (error) {
    return (
      <ErrorAlert 
        title="AI-A計測集計のデータが見つかりません。利用可能なフォルダから選択してください。"
        error={error}
      />
    );
  }

  return (
    <div className="space-y-8">
      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Displacement Chart */}
        <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
          <CardHeader>
            <CardTitle className="text-xl font-semibold text-gray-800 text-center">
              変位量 over TD
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <ChartSkeleton />
            ) : (
              <>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={displacementData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis 
                  dataKey="td" 
                  stroke="#6B7280"
                  label={{ value: 'TD (m)', position: 'insideBottom', offset: -5 }}
                  domain={[0, 1200]}
                />
                <YAxis 
                  stroke="#6B7280"
                  label={{ value: '変位量', angle: -90, position: 'insideLeft' }}
                  domain={[-15, 10]}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #E5E7EB',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }} 
                />
                <Legend />
                <Line type="monotone" dataKey="series3m" stroke="#3B82F6" strokeWidth={2} dot={false} name="3m" />
                <Line type="monotone" dataKey="series5m" stroke="#F59E0B" strokeWidth={2} dot={false} name="5m" />
                <Line type="monotone" dataKey="series10m" stroke="#10B981" strokeWidth={2} dot={false} name="10m" />
                <Line type="monotone" dataKey="series20m" stroke="#EF4444" strokeWidth={2} dot={false} name="20m" />
                <Line type="monotone" dataKey="series50m" stroke="#8B5CF6" strokeWidth={2} dot={false} name="50m" />
                <Line type="monotone" dataKey="series100m" stroke="#6B7280" strokeWidth={2} dot={false} name="100m" />
                  </LineChart>
                </ResponsiveContainer>
                <p className="text-sm text-gray-500 text-center mt-2">変位量の時系列変化</p>
              </>
            )}
          </CardContent>
        </Card>

        {/* Settlement Chart */}
        <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
          <CardHeader>
            <CardTitle className="text-xl font-semibold text-gray-800 text-center">
              沈下量 over TD
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <ChartSkeleton />
            ) : (
              <>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={settlementData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                <XAxis 
                  dataKey="td" 
                  stroke="#6B7280"
                  label={{ value: 'TD (m)', position: 'insideBottom', offset: -5 }}
                  domain={[0, 1200]}
                />
                <YAxis 
                  stroke="#6B7280"
                  label={{ value: '沈下量', angle: -90, position: 'insideLeft' }}
                  domain={[-30, 0]}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #E5E7EB',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }} 
                />
                <Legend />
                <Line type="monotone" dataKey="series3m" stroke="#3B82F6" strokeWidth={2} dot={false} name="3m" />
                <Line type="monotone" dataKey="series5m" stroke="#F59E0B" strokeWidth={2} dot={false} name="5m" />
                <Line type="monotone" dataKey="series10m" stroke="#10B981" strokeWidth={2} dot={false} name="10m" />
                <Line type="monotone" dataKey="series20m" stroke="#EF4444" strokeWidth={2} dot={false} name="20m" />
                <Line type="monotone" dataKey="series50m" stroke="#8B5CF6" strokeWidth={2} dot={false} name="50m" />
                <Line type="monotone" dataKey="series100m" stroke="#6B7280" strokeWidth={2} dot={false} name="100m" />
                  </LineChart>
                </ResponsiveContainer>
                <p className="text-sm text-gray-500 text-center mt-2">沈下量の時系列変化</p>
              </>
            )}
          </CardContent>
        </Card>

        {/* Distribution Chart 1 */}
        <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
          <CardHeader>
            <CardTitle className="text-xl font-semibold text-gray-800 text-center">
              変位量 Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <ChartSkeleton />
            ) : (
              <>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={displacementDistribution} margin={{ top: 5, right: 5, left: 30, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                    <XAxis 
                      dataKey="range" 
                      stroke="#6B7280"
                      label={{ value: '変位量 (mm)', position: 'insideBottom', offset: -5 }}
                      angle={-45}
                      textAnchor="end"
                      height={60}
                      interval={Math.floor(displacementDistribution.length / 20)}
                    />
                    <YAxis 
                      stroke="#6B7280"
                      label={{ value: '頻度', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'white', 
                        border: '1px solid #E5E7EB',
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                      }}
                      content={({ active, payload, label }: any) => {
                        if (active && payload && payload.length > 0) {
                          const data = payload[0].payload;
                          return (
                            <div className="p-2 bg-white rounded shadow">
                              <p className="font-semibold">{label}</p>
                              <p style={{ color: '#3B82F6' }}>3m: {data.series3m || 0}</p>
                              <p style={{ color: '#F59E0B' }}>5m: {data.series5m || 0}</p>
                              <p style={{ color: '#10B981' }}>10m: {data.series10m || 0}</p>
                              <p style={{ color: '#EF4444' }}>20m: {data.series20m || 0}</p>
                              <p style={{ color: '#8B5CF6' }}>50m: {data.series50m || 0}</p>
                              <p style={{ color: '#6B7280' }}>100m: {data.series100m || 0}</p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    {/* ダミーの基準バーにカスタムシェイプを適用して全シリーズを描画 */}
                    <Bar dataKey="dummyBase" shape={CustomOverlayShape} />
                  </BarChart>
                </ResponsiveContainer>
                {/* 凡例 */}
                <div className="flex justify-center mt-4 space-x-4">
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-blue-500 opacity-50 mr-2"></div>
                    <span className="text-sm">3m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-yellow-500 opacity-50 mr-2"></div>
                    <span className="text-sm">5m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-green-500 opacity-50 mr-2"></div>
                    <span className="text-sm">10m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-red-500 opacity-50 mr-2"></div>
                    <span className="text-sm">20m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-purple-500 opacity-50 mr-2"></div>
                    <span className="text-sm">50m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-gray-500 opacity-50 mr-2"></div>
                    <span className="text-sm">100m</span>
                  </div>
                </div>
                <p className="text-sm text-gray-500 text-center mt-2">変位量分布</p>
              </>
            )}
          </CardContent>
        </Card>

        {/* Distribution Chart 2 */}
        <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
          <CardHeader>
            <CardTitle className="text-xl font-semibold text-gray-800 text-center">
              沈下量 Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <ChartSkeleton />
            ) : (
              <>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={settlementDistribution} margin={{ top: 5, right: 5, left: 30, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                    <XAxis 
                      dataKey="range" 
                      stroke="#6B7280"
                      label={{ value: '沈下量 (mm)', position: 'insideBottom', offset: -5 }}
                      angle={-45}
                      textAnchor="end"
                      height={60}
                      interval={Math.floor(settlementDistribution.length / 20)}
                    />
                    <YAxis 
                      stroke="#6B7280"
                      label={{ value: '頻度', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'white', 
                        border: '1px solid #E5E7EB',
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                      }}
                      content={({ active, payload, label }: any) => {
                        if (active && payload && payload.length > 0) {
                          const data = payload[0].payload;
                          return (
                            <div className="p-2 bg-white rounded shadow">
                              <p className="font-semibold">{label}</p>
                              <p style={{ color: '#3B82F6' }}>3m: {data.series3m || 0}</p>
                              <p style={{ color: '#F59E0B' }}>5m: {data.series5m || 0}</p>
                              <p style={{ color: '#10B981' }}>10m: {data.series10m || 0}</p>
                              <p style={{ color: '#EF4444' }}>20m: {data.series20m || 0}</p>
                              <p style={{ color: '#8B5CF6' }}>50m: {data.series50m || 0}</p>
                              <p style={{ color: '#6B7280' }}>100m: {data.series100m || 0}</p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    {/* ダミーの基準バーにカスタムシェイプを適用して全シリーズを描画 */}
                    <Bar dataKey="dummyBase" shape={CustomOverlayShape} />
                  </BarChart>
                </ResponsiveContainer>
                {/* 凡例 */}
                <div className="flex justify-center mt-4 space-x-4">
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-blue-500 opacity-50 mr-2"></div>
                    <span className="text-sm">3m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-yellow-500 opacity-50 mr-2"></div>
                    <span className="text-sm">5m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-green-500 opacity-50 mr-2"></div>
                    <span className="text-sm">10m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-red-500 opacity-50 mr-2"></div>
                    <span className="text-sm">20m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-purple-500 opacity-50 mr-2"></div>
                    <span className="text-sm">50m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-gray-500 opacity-50 mr-2"></div>
                    <span className="text-sm">100m</span>
                  </div>
                </div>
                <p className="text-sm text-gray-500 text-center mt-2">沈下量分布</p>
              </>
            )}
          </CardContent>
        </Card>

        {/* Tunnel Measurement Scatter Plot 1 - 変位量 */}
        <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
          <CardHeader>
            <CardTitle className="text-xl font-semibold text-gray-800 text-center">
              Scatter Plot of 切羽からの距離 vs 計測経過日数 (変位量)
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <ChartSkeleton />
            ) : (
              <>
                <div className="flex items-center">
                  <div className="flex-1">
                    <ResponsiveContainer width="100%" height={300}>
                      <ScatterChart data={convergenceScatterWithColor}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                        <XAxis 
                          type="number"
                          dataKey="x" 
                          stroke="#6B7280" 
                          label={{ value: '切羽からの距離 (m)', position: 'insideBottom', offset: -5 }}
                          domain={[0, 100]}
                        />
                        <YAxis 
                          type="number"
                          dataKey="y" 
                          stroke="#6B7280" 
                          label={{ value: '計測経過日数', angle: -90, position: 'insideLeft' }}
                          domain={[0, 90]}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white', 
                            border: '1px solid #E5E7EB',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }}
                          formatter={(value, name) => [
                            name === 'x' ? `${typeof value === 'number' ? value.toFixed(1) : value}m` : 
                            name === 'y' ? `${typeof value === 'number' ? value.toFixed(1) : value}日` : 
                            name === 'value' ? `${typeof value === 'number' ? value.toFixed(2) : value}mm` :
                            typeof value === 'number' ? value.toFixed(1) : value,
                            name === 'x' ? '距離' : 
                            name === 'y' ? '日数' : 
                            name === 'value' ? '変位量' :
                            '値'
                          ]}
                        />
                        <Scatter 
                          name="散布図" 
                          data={convergenceScatterWithColor} 
                          fill="#3B82F6" 
                          shape={<CustomDot />}
                        >
                          {convergenceScatterWithColor.map((entry, index) => (
                            <Cell key={`cell-conv-${index}`} fill={entry.color} />
                          ))}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                  <ColorBar />
                </div>
                <p className="text-sm text-gray-500 text-center mt-2">変位量による色分け (mm)</p>
              </>
            )}
          </CardContent>
        </Card>

        {/* Tunnel Measurement Scatter Plot 2 - 沈下量 */}
        <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
          <CardHeader>
            <CardTitle className="text-xl font-semibold text-gray-800 text-center">
              Scatter Plot of 切羽からの距離 vs 計測経過日数 (沈下量)
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <ChartSkeleton />
            ) : (
              <>
                <div className="flex items-center">
                  <div className="flex-1">
                    <ResponsiveContainer width="100%" height={300}>
                      <ScatterChart data={settlementScatterWithColor}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                        <XAxis 
                          type="number"
                          dataKey="x" 
                          stroke="#6B7280" 
                          label={{ value: '切羽からの距離 (m)', position: 'insideBottom', offset: -5 }}
                          domain={[0, 100]}
                        />
                        <YAxis 
                          type="number"
                          dataKey="y" 
                          stroke="#6B7280" 
                          label={{ value: '計測経過日数', angle: -90, position: 'insideLeft' }}
                          domain={[0, 90]}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white', 
                            border: '1px solid #E5E7EB',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }}
                          formatter={(value, name) => [
                            name === 'x' ? `${typeof value === 'number' ? value.toFixed(1) : value}m` : 
                            name === 'y' ? `${typeof value === 'number' ? value.toFixed(1) : value}日` : 
                            name === 'value' ? `${typeof value === 'number' ? value.toFixed(2) : value}mm` :
                            typeof value === 'number' ? value.toFixed(1) : value,
                            name === 'x' ? '距離' : 
                            name === 'y' ? '日数' : 
                            name === 'value' ? '沈下量' :
                            '値'
                          ]}
                        />
                        <Scatter 
                          name="散布図" 
                          data={settlementScatterWithColor} 
                          fill="#3B82F6" 
                          shape={<CustomDot />}
                        >
                          {settlementScatterWithColor.map((entry, index) => (
                            <Cell key={`cell-settle-${index}`} fill={entry.color} />
                          ))}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                  <ColorBar />
                </div>
                <p className="text-sm text-gray-500 text-center mt-2">沈下量による色分け (mm)</p>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}