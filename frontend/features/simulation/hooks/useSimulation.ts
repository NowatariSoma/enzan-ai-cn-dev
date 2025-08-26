'use client';

import { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'next/navigation';
import { simulationApi, LocalDisplacementResponse } from '../services/simulationApi';

export function useSimulation() {
  const searchParams = useSearchParams();
  const locationParam = searchParams?.get('location');
  
  const [folders, setFolders] = useState<string[]>([]);
  const [selectedFolder, setSelectedFolder] = useState<string>('');
  const [measurementFiles, setMeasurementFiles] = useState<string[]>([]);
  const [cycleNumber, setCycleNumber] = useState("measurements_A_00004.csv");
  const [distanceFromFace, setDistanceFromFace] = useState(1.0);
  const [excavationAdvance, setExcavationAdvance] = useState(5.0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<LocalDisplacementResponse | null>(null);
  const [chartData, setChartData] = useState<any[]>([]);
  const [predictionChartData, setPredictionChartData] = useState<any[]>([]);
  const [simulationChartData, setSimulationChartData] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Load folders on mount
  useEffect(() => {
    const loadFolders = async () => {
      try {
        const response = await simulationApi.getFolders();
        setFolders(response.folders);
        
        // Set default selected folder - prefer location parameter if valid
        if (response.folders.length > 0) {
          let folderToSelect = response.folders[0]; // Default fallback
          
          if (locationParam && response.folders.includes(locationParam)) {
            folderToSelect = locationParam;
          } else if (locationParam) {
            setError(`指定されたフォルダ '${locationParam}' が見つかりません。利用可能なフォルダから選択してください。`);
          }
          
          setSelectedFolder(folderToSelect);
          console.log('Selected folder:', folderToSelect); // Debug log
        }
      } catch (err) {
        console.error('Failed to load folders:', err);
        setError('フォルダの読み込みに失敗しました。APIサーバーとの接続を確認してください。');
      }
    };
    loadFolders();
  }, [locationParam]);

  // Load measurement files when folder changes
  useEffect(() => {
    if (!selectedFolder) return;
    
    const loadMeasurementFiles = async () => {
      try {
        setError(null); // Clear previous errors
        console.log('Loading measurement files for folder:', selectedFolder); // Debug log
        const response = await simulationApi.getMeasurementFiles(selectedFolder);
        console.log('Received measurement files:', response.measurement_files.length); // Debug log
        setMeasurementFiles(response.measurement_files);
        // Set first file as default
        if (response.measurement_files.length > 0) {
          setCycleNumber(response.measurement_files[0]);
        } else {
          setError(`フォルダ '${selectedFolder}' に測定ファイルが見つかりません。`);
        }
      } catch (err) {
        console.error('Failed to load measurement files:', err);
        const errorMsg = err instanceof Error ? err.message : 'Unknown error';
        setError(`測定ファイルの読み込みに失敗しました: ${errorMsg}`);
      }
    };
    loadMeasurementFiles();
  }, [selectedFolder]);

  const processSimulationData = (data: LocalDisplacementResponse) => {
    // Process the simulation data for charts
    const processedData = data.simulation_data.map(point => {
      const processed: any = {
        distanceFromFace: point.distance_from_face,
      };
      
      // Extract all the displacement and prediction columns
      Object.keys(point).forEach(key => {
        if (key !== 'distance_from_face') {
          // Convert column names to more readable format
          const displayKey = key.replace('_prediction', '予測');
          processed[displayKey] = point[key];
        }
      });
      
      return processed;
    });
    
    return processedData;
  };

  const handleAnalyze = useCallback(async () => {
    if (!selectedFolder || !cycleNumber) {
      setError('フォルダと測定ファイルを選択してください');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      const result = await simulationApi.analyzeLocalDisplacement({
        folder_name: selectedFolder,
        ameasure_file: cycleNumber,
        distance_from_face: distanceFromFace,
        daily_advance: excavationAdvance,
        max_distance_from_face: 200.0,
      });

      setAnalysisResult(result);
      
      // Process and set chart data
      const processedData = processSimulationData(result);
      setChartData(processedData);
      setPredictionChartData(processedData);
      setSimulationChartData(processedData);
      
    } catch (err) {
      console.error('Analysis failed:', err);
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      setError(`局所変位解析に失敗しました: ${errorMsg}`);
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedFolder, cycleNumber, distanceFromFace, excavationAdvance]);

  // Generate chart lines based on actual data columns
  const chartLines = chartData.length > 0 
    ? Object.keys(chartData[0])
        .filter(key => key !== 'distanceFromFace')
        .map((key, index) => {
          const colors = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#06B6D4", "#EF4444", "#EC4899", "#8B5CF6"];
          return {
            dataKey: key,
            stroke: colors[index % colors.length],
            name: key,
            strokeDasharray: key.includes('予測') ? "5 5" : undefined,
          };
        })
    : [];

  return {
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
    setIsAnalyzing,
    chartData,
    predictionChartData,
    simulationChartData,
    handleAnalyze,
    chartLines,
    analysisResult,
    error,
  };
}