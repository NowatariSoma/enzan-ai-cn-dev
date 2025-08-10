'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/layout/card';
import { Skeleton } from '@/components/ui/feedback/skeleton';
import { Alert, AlertDescription } from '@/components/ui/feedback/alert';
import { AlertCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, Cell } from 'recharts';
import { useMeasurementsData } from '../hooks/useMeasurementsData';

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

export function MeasurementsPage() {
  const {
    displacementData,
    settlementData,
    displacementDistribution,
    settlementDistribution,
    scatterData,
    loading,
    error
  } = useMeasurementsData();


  // Show error state
  if (error) {
    return (
      <Alert className="border-red-200 bg-red-50">
        <AlertCircle className="h-4 w-4 text-red-600" />
        <AlertDescription className="text-red-800">
          データの取得に失敗しました。
          <br />
          <span className="text-sm">エラー: {error}</span>
        </AlertDescription>
      </Alert>
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
                <div className="relative" style={{ height: '300px' }}>
                  <div className="absolute inset-0">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={displacementDistribution}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                        <XAxis 
                          dataKey="range" 
                          stroke="#6B7280"
                          label={{ value: '変位量 (mm)', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis 
                          stroke="#6B7280"
                          label={{ value: '頻度', angle: -90, position: 'insideLeft' }}
                          domain={[0, 100]}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white', 
                            border: '1px solid #E5E7EB',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }} 
                        />
                        <Bar dataKey="series3m" fill="#3B82F6" fillOpacity={0.6} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="absolute inset-0">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={displacementDistribution}>
                        <XAxis dataKey="range" stroke="transparent" tick={false} axisLine={false} />
                        <YAxis stroke="transparent" tick={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white', 
                            border: '1px solid #E5E7EB',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }} 
                        />
                        <Bar dataKey="series5m" fill="#F59E0B" fillOpacity={0.6} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="absolute inset-0">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={displacementDistribution}>
                        <XAxis dataKey="range" stroke="transparent" tick={false} axisLine={false} />
                        <YAxis stroke="transparent" tick={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white', 
                            border: '1px solid #E5E7EB',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }} 
                        />
                        <Bar dataKey="series10m" fill="#10B981" fillOpacity={0.6} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="absolute inset-0">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={displacementDistribution}>
                        <XAxis dataKey="range" stroke="transparent" tick={false} axisLine={false} />
                        <YAxis stroke="transparent" tick={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white', 
                            border: '1px solid #E5E7EB',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }} 
                        />
                        <Bar dataKey="series20m" fill="#EF4444" fillOpacity={0.6} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="absolute inset-0">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={displacementDistribution}>
                        <XAxis dataKey="range" stroke="transparent" tick={false} axisLine={false} />
                        <YAxis stroke="transparent" tick={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white', 
                            border: '1px solid #E5E7EB',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }} 
                        />
                        <Bar dataKey="series50m" fill="#8B5CF6" fillOpacity={0.6} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="absolute inset-0">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={displacementDistribution}>
                        <XAxis dataKey="range" stroke="transparent" tick={false} axisLine={false} />
                        <YAxis stroke="transparent" tick={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white', 
                            border: '1px solid #E5E7EB',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }} 
                        />
                        <Bar dataKey="series100m" fill="#6B7280" fillOpacity={0.6} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div className="flex justify-center mt-4 space-x-4">
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-blue-500 opacity-60 mr-2"></div>
                    <span className="text-sm">3m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-yellow-500 opacity-60 mr-2"></div>
                    <span className="text-sm">5m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-green-500 opacity-60 mr-2"></div>
                    <span className="text-sm">10m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-red-500 opacity-60 mr-2"></div>
                    <span className="text-sm">20m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-purple-500 opacity-60 mr-2"></div>
                    <span className="text-sm">50m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-gray-500 opacity-60 mr-2"></div>
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
                <div className="relative" style={{ height: '300px' }}>
                  <div className="absolute inset-0">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={settlementDistribution}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                        <XAxis 
                          dataKey="range" 
                          stroke="#6B7280"
                          label={{ value: '沈下量 (mm)', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis 
                          stroke="#6B7280"
                          label={{ value: '頻度', angle: -90, position: 'insideLeft' }}
                          domain={[0, 100]}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white', 
                            border: '1px solid #E5E7EB',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }} 
                        />
                        <Bar dataKey="series3m" fill="#3B82F6" fillOpacity={0.6} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="absolute inset-0">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={settlementDistribution}>
                        <XAxis dataKey="range" stroke="transparent" tick={false} axisLine={false} />
                        <YAxis stroke="transparent" tick={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white', 
                            border: '1px solid #E5E7EB',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }} 
                        />
                        <Bar dataKey="series5m" fill="#F59E0B" fillOpacity={0.6} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="absolute inset-0">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={settlementDistribution}>
                        <XAxis dataKey="range" stroke="transparent" tick={false} axisLine={false} />
                        <YAxis stroke="transparent" tick={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white', 
                            border: '1px solid #E5E7EB',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }} 
                        />
                        <Bar dataKey="series10m" fill="#10B981" fillOpacity={0.6} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="absolute inset-0">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={settlementDistribution}>
                        <XAxis dataKey="range" stroke="transparent" tick={false} axisLine={false} />
                        <YAxis stroke="transparent" tick={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white', 
                            border: '1px solid #E5E7EB',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }} 
                        />
                        <Bar dataKey="series20m" fill="#EF4444" fillOpacity={0.6} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="absolute inset-0">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={settlementDistribution}>
                        <XAxis dataKey="range" stroke="transparent" tick={false} axisLine={false} />
                        <YAxis stroke="transparent" tick={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white', 
                            border: '1px solid #E5E7EB',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }} 
                        />
                        <Bar dataKey="series50m" fill="#8B5CF6" fillOpacity={0.6} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="absolute inset-0">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={settlementDistribution}>
                        <XAxis dataKey="range" stroke="transparent" tick={false} axisLine={false} />
                        <YAxis stroke="transparent" tick={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white', 
                            border: '1px solid #E5E7EB',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }} 
                        />
                        <Bar dataKey="series100m" fill="#6B7280" fillOpacity={0.6} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div className="flex justify-center mt-4 space-x-4">
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-blue-500 opacity-60 mr-2"></div>
                    <span className="text-sm">3m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-yellow-500 opacity-60 mr-2"></div>
                    <span className="text-sm">5m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-green-500 opacity-60 mr-2"></div>
                    <span className="text-sm">10m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-red-500 opacity-60 mr-2"></div>
                    <span className="text-sm">20m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-purple-500 opacity-60 mr-2"></div>
                    <span className="text-sm">50m</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-4 h-4 bg-gray-500 opacity-60 mr-2"></div>
                    <span className="text-sm">100m</span>
                  </div>
                </div>
                <p className="text-sm text-gray-500 text-center mt-2">沈下量分布</p>
              </>
            )}
          </CardContent>
        </Card>

        {/* Tunnel Measurement Scatter Plot 1 */}
        <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
          <CardHeader>
            <CardTitle className="text-xl font-semibold text-gray-800 text-center">
              Scatter Plot of 切羽からの距離 vs 計測経過日数
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
                      <ScatterChart data={scatterData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                        <XAxis 
                          dataKey="x" 
                          stroke="#6B7280" 
                          label={{ value: '切羽からの距離 (m)', position: 'insideBottom', offset: -5 }}
                          domain={[0, 100]}
                        />
                        <YAxis 
                          dataKey="y" 
                          stroke="#6B7280" 
                          label={{ value: '計測経過日数', angle: -90, position: 'insideLeft' }}
                          domain={[0, 100]}
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
                            typeof value === 'number' ? value.toFixed(1) : value,
                            name === 'x' ? '距離' : 
                            name === 'y' ? '日数' : 
                            '深度'
                          ]}
                        />
                        <Scatter dataKey="y" fill="#3B82F6">
                          {scatterData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                  <ColorBar />
                </div>
                <p className="text-sm text-gray-500 text-center mt-2">トンネル計測データ分布</p>
              </>
            )}
          </CardContent>
        </Card>

        {/* Tunnel Measurement Scatter Plot 2 */}
        <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
          <CardHeader>
            <CardTitle className="text-xl font-semibold text-gray-800 text-center">
              Scatter Plot of 切羽からの距離 vs 計測経過日数
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
                      <ScatterChart data={scatterData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                        <XAxis 
                          dataKey="x" 
                          stroke="#6B7280" 
                          label={{ value: '切羽からの距離 (m)', position: 'insideBottom', offset: -5 }}
                          domain={[0, 100]}
                        />
                        <YAxis 
                          dataKey="y" 
                          stroke="#6B7280" 
                          label={{ value: '計測経過日数', angle: -90, position: 'insideLeft' }}
                          domain={[0, 100]}
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
                            typeof value === 'number' ? value.toFixed(1) : value,
                            name === 'x' ? '距離' : 
                            name === 'y' ? '日数' : 
                            '深度'
                          ]}
                        />
                        <Scatter dataKey="y" fill="#3B82F6">
                          {scatterData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                  <ColorBar />
                </div>
                <p className="text-sm text-gray-500 text-center mt-2">トンネル計測データ分布</p>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}