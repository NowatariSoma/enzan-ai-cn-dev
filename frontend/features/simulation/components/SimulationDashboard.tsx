"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/layout/card";
import { Button } from "@/components/ui/forms/button";
import { Input } from "@/components/ui/inputs/input";
import { Label } from "@/components/ui/inputs/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/inputs/select";
import { ChartsSection } from "./ChartsSection";
import { PredictionDataTable } from "./PredictionDataTable";
import { usePredictionData, useSimulation } from "../hooks";
import { Calculator, TrendingUp, Activity } from "lucide-react";

export function SimulationDashboard() {
  const {
    cycleNumber,
    setCycleNumber,
    distanceFromFace,
    setDistanceFromFace,
    excavationAdvance,
    setExcavationAdvance,
    isAnalyzing,
    chartData,
    handleAnalyze,
    chartLines
  } = useSimulation();

  const { predictionData } = usePredictionData(excavationAdvance, distanceFromFace);


  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <Card className="shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calculator className="h-5 w-5" />
            解析パラメータ
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-row gap-6">
            <div className="flex-1 space-y-2">
              <Label htmlFor="cycle-select" className="text-sm font-medium text-gray-700">
                サイクル選択
              </Label>
              <Select value={cycleNumber} onValueChange={setCycleNumber}>
                <SelectTrigger id="cycle-select" className="w-full">
                  <SelectValue placeholder="測定ファイルを選択してください" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="measurements_A_00004.csv">mesuarments_A-00004.csv</SelectItem>
                  <SelectItem value="measurements_A_00005.csv">mesuarments_A-00005.csv</SelectItem>
                  <SelectItem value="measurements_A_00006.csv">mesuarments_A-00006.csv</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex-1 space-y-2">
              <Label htmlFor="distance" className="text-sm font-medium text-gray-700">
                切羽からの距離 (m)
              </Label>
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setDistanceFromFace(Math.max(0.1, distanceFromFace - 0.1))}
                  className="w-8 h-8 p-0 text-lg font-medium"
                >
                  −
                </Button>
                <Input
                  id="distance"
                  type="number"
                  value={distanceFromFace}
                  onChange={(e) => setDistanceFromFace(parseFloat(e.target.value) || 0)}
                  step="0.1"
                  min="0.1"
                  className="text-center font-medium"
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setDistanceFromFace(distanceFromFace + 0.1)}
                  className="w-8 h-8 p-0 text-lg font-medium"
                >
                  ＋
                </Button>
              </div>
            </div>

            <div className="flex-1 space-y-2">
              <Label htmlFor="excavation" className="text-sm font-medium text-gray-700">
                日進掘進量 (m/日)
              </Label>
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setExcavationAdvance(Math.max(0.1, excavationAdvance - 0.1))}
                  className="w-8 h-8 p-0 text-lg font-medium"
                >
                  −
                </Button>
                <Input
                  id="excavation"
                  type="number"
                  value={excavationAdvance}
                  onChange={(e) => setExcavationAdvance(parseFloat(e.target.value) || 0)}
                  step="0.1"
                  min="0.1"
                  className="text-center font-medium"
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setExcavationAdvance(excavationAdvance + 0.1)}
                  className="w-8 h-8 p-0 text-lg font-medium"
                >
                  ＋
                </Button>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-4 flex justify-center">
            <Button 
              onClick={handleAnalyze} 
              className="w-full md:w-auto px-8 py-2"
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <Activity className="mr-2 h-4 w-4 animate-spin" />
                  解析中...
                </>
              ) : (
                <>
                  <TrendingUp className="mr-2 h-4 w-4" />
                  局所変位解析を実行
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Charts */}
      <ChartsSection
        title="Simulation (5.0 m/day from TD: 1.0m)"
        chartData={chartData}
        chartLines={chartLines}
      />

      {/* Simulation Results */}
      <ChartsSection
        title="Simulation (5.0 m/day from TD: 2.0m)"
        chartData={chartData}
        chartLines={chartLines}
      />

      {/* Prediction Data Table */}
      <PredictionDataTable
        excavationAdvance={excavationAdvance}
        distanceFromFace={distanceFromFace}
        data={predictionData}
      />
    </div>
  );
} 