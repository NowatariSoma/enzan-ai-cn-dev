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
  const [simulationChartData, setSimulationChartData] = useState<any>({});
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
        distanceFromFace: point.distance_from_face || point['切羽からの距離'],
      };
      
      // Extract all the displacement and prediction columns
      Object.keys(point).forEach(key => {
        if (key !== 'distance_from_face' && key !== '切羽からの距離') {
          // Convert column names to more readable format
          const displayKey = key.replace('_prediction', '予測');
          processed[displayKey] = point[key];
        }
      });
      
      return processed;
    });
    
    return processedData;
  };

  const processPredictionData = (data: LocalDisplacementResponse) => {
    // Process the prediction data for charts (actual measurements + predictions)
    if (!data.prediction_data) {
      return [];
    }
    
    const processedData = data.prediction_data.map(point => {
      const processed: any = {
        distanceFromFace: point.distance_from_face || point['切羽からの距離'],
      };
      
      // Extract all columns (actual measurements and predictions)
      Object.keys(point).forEach(key => {
        if (key !== 'distance_from_face' && key !== '切羽からの距離') {
          // Convert column names to more readable format
          let displayKey = key;
          if (key.includes('_prediction')) {
            displayKey = key.replace('_prediction', '予測');
          }
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
      const simulationData = processSimulationData(result);
      const predictionData = processPredictionData(result);
      
      console.log('🔍 DEBUG - simulationData length:', simulationData.length);
      console.log('🔍 DEBUG - predictionData length:', predictionData.length);
      if (simulationData.length > 0) {
        console.log('🔍 DEBUG - first simulationData:', simulationData[0]);
        console.log('🔍 DEBUG - last simulationData:', simulationData[simulationData.length - 1]);
      }
      
      // For simulation charts, we need both actual measurements and simulation predictions
      // Create data series for MultiLineChart
      console.log('🔍 DEBUG - predictionData columns:', Object.keys(predictionData[0] || {}));
      console.log('🔍 DEBUG - simulationData columns:', Object.keys(simulationData[0] || {}));
      
      const actualColumns = Object.keys(predictionData[0] || {}).filter(
        key => key !== 'distanceFromFace' && key !== '切羽からの距離' && 
               !key.includes('予測') && !key.includes('prediction')
      );
      
      const simulationColumns = Object.keys(simulationData[0] || {}).filter(
        key => key !== 'distanceFromFace' && key !== '切羽からの距離' &&
               (key.includes('予測') || key.includes('prediction'))
      );
      
      console.log('🔍 DEBUG - actualColumns:', actualColumns);
      console.log('🔍 DEBUG - simulationColumns:', simulationColumns);

      // Create separate data series for displacement and settlement
      const displacementColumns = actualColumns.filter(col => 
        col.includes('変位') || col.includes('displacement') || 
        (col.match(/[ABC]/) && !col.includes('沈下') && !col.includes('settlement'))
      );
      
      const settlementColumns = actualColumns.filter(col => 
        col.includes('沈下') || col.includes('settlement')
      );
      
      const displacementPredictionColumns = simulationColumns.filter(col => 
        col.includes('変位') || col.includes('displacement') || 
        (col.match(/[ABC]/) && !col.includes('沈下') && !col.includes('settlement'))
      );
      
      const settlementPredictionColumns = simulationColumns.filter(col => 
        col.includes('沈下') || col.includes('settlement')
      );

      const displacementDataSeries = [
        {
          name: "実測",
          data: predictionData,
          color: "#3B82F6",
          strokeDasharray: "5 5",
          columns: displacementColumns
        },
        {
          name: "予測",
          data: simulationData,
          color: "#F59E0B",
          columns: displacementPredictionColumns
        }
      ];

      const settlementDataSeries = [
        {
          name: "実測",
          data: predictionData,
          color: "#3B82F6",
          strokeDasharray: "5 5",
          columns: settlementColumns
        },
        {
          name: "予測",
          data: simulationData,
          color: "#F59E0B",
          columns: settlementPredictionColumns
        }
      ];

      setChartData(simulationData);
      setPredictionChartData(predictionData);
      setSimulationChartData({
        displacement: displacementDataSeries,
        settlement: settlementDataSeries
      });
      
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
        .filter(key => key !== 'distanceFromFace' && key !== '切羽からの距離')
        .map((key, index) => {
          const colors = ["#3B82F6", "#F59E0B", "#10B981", "#8B5CF6", "#06B6D4", "#EF4444", "#EC4899", "#8B5CF6"];
          // Determine color based on A, B, C pattern
          let colorIndex = index % colors.length;
          if (key.includes('A')) colorIndex = 0;
          else if (key.includes('B')) colorIndex = 1;
          else if (key.includes('C')) colorIndex = 2;
          
          return {
            dataKey: key,
            stroke: colors[colorIndex],
            name: key,
            strokeDasharray: (key.includes('予測') || key.includes('prediction')) ? undefined : "5 5",
          };
        })
    : [];

  // Generate specific chart lines for displacement data (最終変位量) - use simulationChartData for simulation charts
  const displacementChartLines = predictionChartData.length > 0 
    ? Object.keys(predictionChartData[0])
        .filter(key => key !== 'distanceFromFace' && key !== '切羽からの距離' && 
                (key.includes('変位') || key.includes('displacement') || 
                 (key.match(/[ABC]/) && !key.includes('沈下') && !key.includes('settlement'))))
        .map((key, index) => {
          const colors = ["#3B82F6", "#F59E0B", "#10B981", "#8B5CF6"];
          // Determine color based on A, B, C pattern
          let colorIndex = index % colors.length;
          if (key.includes('A')) colorIndex = 0;
          else if (key.includes('B')) colorIndex = 1;
          else if (key.includes('C')) colorIndex = 2;
          
          return {
            dataKey: key,
            stroke: colors[colorIndex],
            name: key,
            strokeDasharray: (key.includes('予測') || key.includes('prediction')) ? undefined : "5 5",
          };
        })
    : [];

  // Generate chart lines for displacement simulation data - separate lines for actual and simulation data
  const simulationDisplacementChartLines = simulationChartData.length > 0 
    ? Object.keys(simulationChartData[0])
        .filter(key => key !== 'distanceFromFace' && key !== '切羽からの距離' &&
                (key.includes('変位') || key.includes('displacement') || 
                 (key.match(/[ABC]/) && !key.includes('沈下') && !key.includes('settlement'))))
        .map((key) => {
          const colors = ["#3B82F6", "#F59E0B", "#10B981", "#8B5CF6"];
          // Determine color based on A, B, C pattern
          let colorIndex = 0;
          if (key.includes('A')) colorIndex = 0;
          else if (key.includes('B')) colorIndex = 1;
          else if (key.includes('C')) colorIndex = 2;
          
          // Different styling for actual measurements vs simulation predictions
          const isSimulation = key.includes('_sim');
          const displayName = key.replace('_sim', '') + (isSimulation ? ' (シミュレーション)' : ' (実測)');
          
          return {
            dataKey: key,
            stroke: colors[colorIndex],
            name: displayName,
            strokeDasharray: isSimulation ? undefined : "5 5", // Solid line for simulation, dashed for actual
          };
        })
    : [];

  // Generate specific chart lines for settlement data (最終沈下量)
  const settlementChartLines = predictionChartData.length > 0 
    ? Object.keys(predictionChartData[0])
        .filter(key => key !== 'distanceFromFace' && key !== '切羽からの距離' && 
                (key.includes('沈下') || key.includes('settlement') || 
                 (key.match(/[ABC]/) && (key.includes('沈下') || key.includes('settlement')))))
        .map((key, index) => {
          const colors = ["#3B82F6", "#F59E0B", "#10B981", "#8B5CF6"];
          // Determine color based on settlement type (沈下量1, 沈下量2, 沈下量3)
          let colorIndex = 0;
          if (key.includes('沈下量1') || key.includes('settlement1')) colorIndex = 0;
          else if (key.includes('沈下量2') || key.includes('settlement2')) colorIndex = 1;
          else if (key.includes('沈下量3') || key.includes('settlement3')) colorIndex = 2;
          else if (key.includes('1')) colorIndex = 0;
          else if (key.includes('2')) colorIndex = 1;
          else if (key.includes('3')) colorIndex = 2;
          
          return {
            dataKey: key,
            stroke: colors[colorIndex],
            name: key,
            strokeDasharray: (key.includes('予測') || key.includes('prediction')) ? undefined : "5 5",
          };
        })
    : [];

  // Generate chart lines for settlement simulation data - separate lines for actual and simulation data
  const simulationSettlementChartLines = simulationChartData.length > 0 
    ? Object.keys(simulationChartData[0])
        .filter(key => key !== 'distanceFromFace' && key !== '切羽からの距離' &&
                (key.includes('沈下') || key.includes('settlement') || 
                 (key.match(/[ABC]/) && (key.includes('沈下') || key.includes('settlement')))))
        .map((key) => {
          const colors = ["#3B82F6", "#F59E0B", "#10B981", "#8B5CF6"];
          // Determine color based on settlement type (沈下量1, 沈下量2, 沈下量3)
          let colorIndex = 0;
          if (key.includes('沈下量1') || key.includes('settlement1')) colorIndex = 0;
          else if (key.includes('沈下量2') || key.includes('settlement2')) colorIndex = 1;
          else if (key.includes('沈下量3') || key.includes('settlement3')) colorIndex = 2;
          else if (key.includes('1')) colorIndex = 0;
          else if (key.includes('2')) colorIndex = 1;
          else if (key.includes('3')) colorIndex = 2;
          
          // Different styling for actual measurements vs simulation predictions
          const isSimulation = key.includes('_sim');
          const displayName = key.replace('_sim', '') + (isSimulation ? ' (シミュレーション)' : ' (実測)');
          
          return {
            dataKey: key,
            stroke: colors[colorIndex],
            name: displayName,
            strokeDasharray: isSimulation ? undefined : "5 5", // Solid line for simulation, dashed for actual
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
    displacementChartLines,
    settlementChartLines,
    simulationDisplacementChartLines,
    simulationSettlementChartLines,
    analysisResult,
    error,
  };
}