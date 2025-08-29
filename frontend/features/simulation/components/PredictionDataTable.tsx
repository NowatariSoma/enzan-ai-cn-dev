"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/layout/card";
import { DataTable } from "@/components/ui/data-display/data-table";

interface PredictionDataTableProps {
  excavationAdvance: number;
  distanceFromFace: number;
  data: Array<{
    distance_from_face: number;
    [key: string]: number;
  }>;
}

export function PredictionDataTable({
  excavationAdvance,
  distanceFromFace,
  data,
}: PredictionDataTableProps) {
  // Normalize data - convert Japanese key to English key if it exists
  const normalizedData = data.map(item => {
    const normalized = { ...item };
    if ('切羽からの距離' in item) {
      normalized.distance_from_face = item['切羽からの距離'];
      delete normalized['切羽からの距離'];
    }
    return normalized;
  });
  // Fixed columns to match learning page format
  const tableColumns = [
    { key: "distance_from_face" as const, header: "切羽からの距離 (m)" },
    ...(normalizedData.length > 0 ? Object.keys(normalizedData[0])
      .filter(key => key !== 'distance_from_face' && key !== '切羽からの距離' && key !== 'distanceFromFace')
      .slice(0, 6) // Limit to first 6 columns for readability
      .map(key => ({
        key: key as const,
        header: key.replace('_', ' '),
      })) : [])
  ];

  return (
    <Card className="shadow-md">
      <CardHeader>
        <CardTitle className="text-center">
          予測データテーブル
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <DataTable
            data={normalizedData}
            columns={tableColumns}
            itemsPerPage={8}
          />
        </div>
      </CardContent>
    </Card>
  );
}