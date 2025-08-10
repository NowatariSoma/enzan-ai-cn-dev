import type { WorkerCount, HistoricalData, AggregatedData } from '@/types/worker';

export function aggregateData(
  data: HistoricalData[],
  intervalMinutes: number
): AggregatedData[] {
  if (data.length === 0) return [];

  const intervals = new Map<string, HistoricalData[]>();
  const intervalMs = intervalMinutes * 60 * 1000;

  data.forEach((item) => {
    const timestamp = new Date(item.createdAt).getTime();
    const intervalStart = Math.floor(timestamp / intervalMs) * intervalMs;
    const intervalKey = new Date(intervalStart).toISOString();

    if (!intervals.has(intervalKey)) {
      intervals.set(intervalKey, []);
    }
    intervals.get(intervalKey)!.push(item);
  });

  return Array.from(intervals.entries()).map(([timeRange, items]) => {
    const counts = items.map(item => item.counts);
    
    return {
      timeRange,
      totalReadings: items.length,
      averageCounts: {
        red: Math.round(counts.reduce((sum, c) => sum + c.red, 0) / counts.length),
        other: Math.round(counts.reduce((sum, c) => sum + c.other, 0) / counts.length),
        no_helmet: Math.round(counts.reduce((sum, c) => sum + c.no_helmet, 0) / counts.length),
        total: Math.round(counts.reduce((sum, c) => sum + c.total, 0) / counts.length),
        timestamp: timeRange,
      },
      maxCounts: {
        red: Math.max(...counts.map(c => c.red)),
        other: Math.max(...counts.map(c => c.other)),
        no_helmet: Math.max(...counts.map(c => c.no_helmet)),
        total: Math.max(...counts.map(c => c.total)),
        timestamp: timeRange,
      },
      minCounts: {
        red: Math.min(...counts.map(c => c.red)),
        other: Math.min(...counts.map(c => c.other)),
        no_helmet: Math.min(...counts.map(c => c.no_helmet)),
        total: Math.min(...counts.map(c => c.total)),
        timestamp: timeRange,
      },
    };
  }).sort((a, b) => new Date(a.timeRange).getTime() - new Date(b.timeRange).getTime());
}

export function exportToCSV(data: AggregatedData[], filename: string) {
  const headers = [
    'Time Range',
    'Total Readings',
    'Avg Red Helmets',
    'Avg Other Helmets',
    'Avg No Helmets',
    'Avg Total',
    'Max Red Helmets',
    'Max Other Helmets',
    'Max No Helmets',
    'Max Total',
    'Min Red Helmets',
    'Min Other Helmets',
    'Min No Helmets',
    'Min Total',
  ];

  const csvContent = [
    headers.join(','),
    ...data.map(item => [
      new Date(item.timeRange).toLocaleString(),
      item.totalReadings,
      item.averageCounts.red,
      item.averageCounts.other,
      item.averageCounts.no_helmet,
      item.averageCounts.total,
      item.maxCounts.red,
      item.maxCounts.other,
      item.maxCounts.no_helmet,
      item.maxCounts.total,
      item.minCounts.red,
      item.minCounts.other,
      item.minCounts.no_helmet,
      item.minCounts.total,
    ].join(','))
  ].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}