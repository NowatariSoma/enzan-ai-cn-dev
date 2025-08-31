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
  // Fixed columns to display specific prediction columns
  const tableColumns = [
    { key: "distance_from_face" as const, header: "切羽からの距離" },
    { key: "変位量A_prediction" as const, header: "変位量A_prediction" },
    { key: "変位量B_prediction" as const, header: "変位量B_prediction" },
    { key: "変位量C_prediction" as const, header: "変位量C_prediction" },
    { key: "沈下量1_prediction" as const, header: "沈下量1_prediction" },
    { key: "沈下量2_prediction" as const, header: "沈下量2_prediction" },
    { key: "沈下量3_prediction" as const, header: "沈下量3_prediction" }
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