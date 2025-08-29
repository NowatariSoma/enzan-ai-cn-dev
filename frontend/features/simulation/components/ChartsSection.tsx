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
        <PredictionChart
          data={chartData}
          title={title}
          xAxisLabel="切羽からの距離 (m)"
          yAxisLabel="変位量 (mm)"
          lines={chartLines}
        />
      </CardContent>
    </Card>
  );
}