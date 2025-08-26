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
import { ScatterPlotSection } from "./ScatterPlotSection";
import { FeatureImportanceSection } from "./FeatureImportanceSection";
import { HeatmapSection } from "./HeatmapSection";
import { PredictionDataTable } from "./PredictionDataTable";
import { usePredictionData, useLearning, useHeatmap } from "../hooks";
import { Calculator, TrendingUp, Activity } from "lucide-react";

export function LearningDashboard() {
  const {
    folderName,
    model,
    setModel,
    dataType,
    setDataType,
    predictionTD,
    setPredictionTD,
    maxDistance,
    setMaxDistance,
    isAnalyzing,
    trainScatterDataA,
    trainScatterDataB,
    validationScatterDataA,
    validationScatterDataB,
    featureImportanceA,
    featureImportanceB,
    heatmapDataA,
    heatmapDataB,
    trainRSquaredA,
    trainRSquaredB,
    validationRSquaredA,
    validationRSquaredB,
    trainMSEA,
    trainMSEB,
    validationMSEA,
    validationMSEB,
    handleAnalyze,
    processEachData,
  } = useLearning();

  // Debug logging
  console.log('Dashboard render:', {
    hasProcessEachData: !!processEachData,
    trainScatterDataCount: trainScatterDataA?.length || 0,
    validationScatterDataCount: validationScatterDataA?.length || 0,
    trainMSEA,
    trainRSquaredA,
    isAnalyzing,
  });

  const { loading: heatmapLoading, error: heatmapError } = useHeatmap();

  const { predictionData } = usePredictionData(maxDistance, predictionTD);

  return (
    <div className="space-y-8">
      {/* Control Panel */}
      <Card className="shadow-lg hover:shadow-xl transition-shadow duration-300">
        <CardHeader>
          <CardTitle className="text-xl font-semibold text-gray-800 flex items-center gap-2">
            <Calculator className="h-5 w-5" />
            解析パラメータ
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

            <div className="space-y-2">
              <Label htmlFor="model-select" className="text-sm font-medium text-gray-700">
                Select Model
              </Label>
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger id="model-select" className="w-full">
                  <SelectValue placeholder="モデルを選択してください" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Random Forest">Random Forest</SelectItem>
                  <SelectItem value="Linear Regression">Linear Regression</SelectItem>
                  <SelectItem value="SVR">SVR</SelectItem>
                  <SelectItem value="HistGradientBoostingRegressor">Hist Gradient Boosting</SelectItem>
                  <SelectItem value="MLP">MLP Neural Network</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="data-type-select" className="text-sm font-medium text-gray-700">
                Data Type
              </Label>
              <Select value={dataType} onValueChange={(value: 'settlement' | 'convergence') => setDataType(value)}>
                <SelectTrigger id="data-type-select" className="w-full">
                  <SelectValue placeholder="データタイプを選択してください" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="settlement">Settlement (沈下量)</SelectItem>
                  <SelectItem value="convergence">Convergence (変位量)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="prediction-td" className="text-sm font-medium text-gray-700">
                prediction TD(m)
              </Label>
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPredictionTD(Math.max(10, predictionTD - 10))}
                  className="w-8 h-8 p-0 text-lg font-medium"
                >
                  −
                </Button>
                <Input
                  id="prediction-td"
                  type="number"
                  value={predictionTD}
                  onChange={(e) => setPredictionTD(parseInt(e.target.value) || 0)}
                  step="10"
                  min="10"
                  className="text-center font-medium"
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPredictionTD(predictionTD + 10)}
                  className="w-8 h-8 p-0 text-lg font-medium"
                >
                  ＋
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="max-distance" className="text-sm font-medium text-gray-700">
                Max distance from cutter face
              </Label>
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setMaxDistance(Math.max(10, maxDistance - 10))}
                  className="w-8 h-8 p-0 text-lg font-medium"
                >
                  −
                </Button>
                <Input
                  id="max-distance"
                  type="number"
                  value={maxDistance}
                  onChange={(e) => setMaxDistance(parseInt(e.target.value) || 0)}
                  step="10"
                  min="10"
                  className="text-center font-medium"
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setMaxDistance(maxDistance + 10)}
                  className="w-8 h-8 p-0 text-lg font-medium"
                >
                  ＋
                </Button>
              </div>
            </div>
          </div>

          <div className="mt-6 pt-4 flex justify-center">
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
                  予測モデル作成を実行
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Train Data Scatter Plots */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Train Data</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <ScatterPlotSection
            title={`Actual vs Predicted for ${dataType === 'settlement' ? '沈下量' : '変位量'}A (Train Data)`}
            data={trainScatterDataA}
            rSquared={trainRSquaredA}
            mse={trainMSEA}
            xLabel={`Actual ${dataType === 'settlement' ? '沈下量' : '変位量'}A`}
            yLabel={`Predicted ${dataType === 'settlement' ? '沈下量' : '変位量'}A`}
          />
          
          <ScatterPlotSection
            title={`Actual vs Predicted for ${dataType === 'settlement' ? '沈下量' : '変位量'}B (Train Data)`}
            data={trainScatterDataB}
            rSquared={trainRSquaredB}
            mse={trainMSEB}
            xLabel={`Actual ${dataType === 'settlement' ? '沈下量' : '変位量'}B`}
            yLabel={`Predicted ${dataType === 'settlement' ? '沈下量' : '変位量'}B`}
          />
        </div>
      </div>

      {/* Validation Data Scatter Plots */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Validation Data</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <ScatterPlotSection
            title={`Actual vs Predicted for ${dataType === 'settlement' ? '沈下量' : '変位量'}A (Validation Data)`}
            data={validationScatterDataA}
            rSquared={validationRSquaredA}
            mse={validationMSEA}
            xLabel={`Actual ${dataType === 'settlement' ? '沈下量' : '変位量'}A`}
            yLabel={`Predicted ${dataType === 'settlement' ? '沈下量' : '変位量'}A`}
          />
          
          <ScatterPlotSection
            title={`Actual vs Predicted for ${dataType === 'settlement' ? '沈下量' : '変位量'}B (Validation Data)`}
            data={validationScatterDataB}
            rSquared={validationRSquaredB}
            mse={validationMSEB}
            xLabel={`Actual ${dataType === 'settlement' ? '沈下量' : '変位量'}B`}
            yLabel={`Predicted ${dataType === 'settlement' ? '沈下量' : '変位量'}B`}
          />
        </div>
      </div>

      {/* Feature Importance */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Feature Importance</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <FeatureImportanceSection
            title={`Feature Importance for ${dataType === 'settlement' ? '沈下量' : '変位量'}A`}
            data={featureImportanceA}
          />
          
          <FeatureImportanceSection
            title={`Feature Importance for ${dataType === 'settlement' ? '沈下量' : '変位量'}B`}
            data={featureImportanceB}
          />
        </div>
      </div>

      {/* Heatmap */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Heatmap</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <HeatmapSection
            title="Heatmap"
            data={heatmapDataA.data}
            features={heatmapDataA.features}
            loading={heatmapLoading}
            error={heatmapError}
          />
          
          <HeatmapSection
            title="Heatmap"
            data={heatmapDataB.data}
            features={heatmapDataB.features}
            loading={heatmapLoading}
            error={heatmapError}
          />
        </div>
      </div>
    </div>
  );
} 