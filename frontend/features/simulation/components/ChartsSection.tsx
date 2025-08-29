"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/layout/card";
import { PredictionChart } from "./PredictionChart";

interface DataSeries {
  name: string;
  data: any[];
  color: string;
  strokeDasharray?: string;
  columns: string[];
}

interface ChartsSectionProps {
  title: string;
  chartData?: any[]; // For single dataset (backward compatibility)
  chartLines?: Array<{
    dataKey: string;
    stroke: string;
    name: string;
    strokeDasharray?: string;
  }>;
  // For multiple datasets
  dataSeries?: DataSeries[];
  xAxisLabel?: string;
  yAxisLabel?: string;
}

export function ChartsSection({
  title,
  chartData,
  chartLines,
  dataSeries,
  xAxisLabel = "切羽からの距離 (m)",
  yAxisLabel = "変位量 (mm)",
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
          data={chartData || []}
          title={title}
          xAxisLabel={xAxisLabel}
          yAxisLabel={yAxisLabel}
          lines={chartLines || []}
          dataSeries={dataSeries}
        />
      </CardContent>
    </Card>
  );
}