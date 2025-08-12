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
import { Alert, AlertDescription } from "@/components/ui/feedback/alert";
import { ChartsSection } from "./ChartsSection";
import { PredictionDataTable } from "./PredictionDataTable";
import { usePredictionData, useSimulation } from "../hooks";
import { Calculator, TrendingUp, Activity, AlertCircle } from "lucide-react";

export function SimulationDashboard() {
  const {
    folders,
    selectedFolder,
    setSelectedFolder,
    measurementFiles,
    cycleNumber,
    setCycleNumber,
    distanceFromFace,
    setDistanceFromFace,
    excavationAdvance,
    setExcavationAdvance,
    isAnalyzing,
    chartData,
    predictionChartData,
    simulationChartData,
    handleAnalyze,
    chartLines,
    analysisResult,
    error
  } = useSimulation();

  const { predictionData } = usePredictionData(excavationAdvance, distanceFromFace);


  return (
    <div className="space-y-6">
      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Control Panel */}
      <Card className="shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calculator className="h-5 w-5" />
            解析パラメータ
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Folder Selection */}
            <div className="space-y-2">
              <Label htmlFor="folder-select" className="text-sm font-medium text-gray-700">
                フォルダ選択
              </Label>
              <Select value={selectedFolder} onValueChange={setSelectedFolder}>
                <SelectTrigger id="folder-select" className="w-full">
                  <SelectValue placeholder="フォルダを選択してください" />
                </SelectTrigger>
                <SelectContent>
                  {folders.map(folder => (
                    <SelectItem key={folder} value={folder}>
                      {folder}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Parameters Row */}
            <div className="flex flex-row gap-6">
              <div className="flex-1 space-y-2">
                <Label htmlFor="cycle-select" className="text-sm font-medium text-gray-700">
                  測定ファイル選択
                </Label>
                <Select value={cycleNumber} onValueChange={setCycleNumber}>
                  <SelectTrigger id="cycle-select" className="w-full">
                    <SelectValue placeholder="測定ファイルを選択してください" />
                  </SelectTrigger>
                  <SelectContent>
                    {measurementFiles.map(file => (
                      <SelectItem key={file} value={file}>
                        {file}
                      </SelectItem>
                    ))}
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
                disabled={isAnalyzing || !selectedFolder || !cycleNumber}
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
          </div>
        </CardContent>
      </Card>

      {/* Display Results Only After Analysis */}
      {analysisResult && (
        <>
          {/* Prediction Charts */}
          <ChartsSection
            title={`Prediction (Actual excavation) - Cycle ${analysisResult.cycle_no} (TD: ${analysisResult.td})`}
            chartData={predictionChartData}
            chartLines={chartLines}
          />

          {/* Simulation Charts */}
          <ChartsSection
            title={`Simulation (${excavationAdvance} m/day from TD: ${distanceFromFace}m)`}
            chartData={simulationChartData}
            chartLines={chartLines}
          />

          {/* Prediction Data Table */}
          <PredictionDataTable
            excavationAdvance={excavationAdvance}
            distanceFromFace={distanceFromFace}
            data={analysisResult.simulation_data}
          />
        </>
      )}
    </div>
  );
} 