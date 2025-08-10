"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/layout/card";
import { PredictionChart } from "./PredictionChart";

interface ChartsSectionProps {
  title: string;
  chartData: any[];
  chartLines: Array<{
    dataKey: string;
    stroke: string;
    name: string;
    strokeDasharray?: string;
  }>;
}

export function ChartsSection({
  title,
  chartData,
  chartLines,
}: ChartsSectionProps) {
  return (
    <Card className="shadow-md">
      <CardHeader>
        <CardTitle className="text-center">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <PredictionChart
            data={chartData}
            title="変位量データ (A, B, C)"
            xAxisLabel="切羽からの距離 (m)"
            yAxisLabel="変位量 (mm)"
            lines={chartLines}
          />
          <PredictionChart
            data={chartData}
            title="予測データ (A_prediction, B_prediction, C_prediction)"
            xAxisLabel="切羽からの距離 (m)"
            yAxisLabel="変位量 (mm)"
            lines={chartLines}
          />
        </div>
      </CardContent>
    </Card>
  );
}