"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/layout/card";
import { DataTable } from "@/components/ui/data-display/data-table";

interface PredictionDataTableProps {
  excavationAdvance: number;
  distanceFromFace: number;
  data: Array<{
    切羽からの距離?: number;
    distance_from_face?: number;
    [key: string]: number;
  }>;
}

export function PredictionDataTable({
  excavationAdvance,
  distanceFromFace,
  data,
}: PredictionDataTableProps) {
  // Dynamically generate columns based on actual data
  const tableColumns = data.length > 0 ? [
    { 
      key: (data[0].切羽からの距離 !== undefined ? "切羽からの距離" : "distance_from_face") as const, 
      header: "切羽からの距離 (m)" 
    },
    ...Object.keys(data[0])
      .filter(key => key !== 'distance_from_face' && key !== '切羽からの距離')
      .slice(0, 6) // Limit to first 6 columns for readability
      .map(key => ({
        key: key as const,
        header: key.replace('_', ' '),
      }))
  ] : [
    { key: "distance_from_face" as const, header: "切羽からの距離 (m)" },
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
            data={data}
            columns={tableColumns}
            itemsPerPage={8}
          />
        </div>
      </CardContent>
    </Card>
  );
}