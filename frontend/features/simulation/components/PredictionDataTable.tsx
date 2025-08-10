"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/layout/card";
import { DataTable } from "@/components/ui/data-display/data-table";

interface PredictionDataTableProps {
  excavationAdvance: number;
  distanceFromFace: number;
  data: Array<{
    step: number;
    days: number;
    prediction1: string;
    prediction2: string;
    prediction3: string;
  }>;
}

export function PredictionDataTable({
  excavationAdvance,
  distanceFromFace,
  data,
}: PredictionDataTableProps) {
  const tableColumns = [
    { key: "step" as const, header: "切羽からの距離" },
    { key: "days" as const, header: "変位量A_prediction" },
    { key: "prediction1" as const, header: "変位量B_prediction" },
    { key: "prediction2" as const, header: "変位量C_prediction" },
    { key: "prediction3" as const, header: "沈下量1_prediction" },
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