'use client';

import { useEffect } from 'react';
import { CheckCircle, Info } from 'lucide-react';
import { 
  StepDefinition, 
  StepStatus, 
  GenericStepNavigationProps 
} from '../../types/navigation';

export function StepNavigation<T extends string>({ 
  steps,
  currentStep,
  onStepChange,
  canNavigateToStep = () => true,
  getNavigationMessage = (stepKey) => `${steps.find(s => s.key === stepKey)?.label}に移動`,
  className = '',
  showProgressBar = true,
  showCurrentStepInfo = true,
  debugMode = false
}: GenericStepNavigationProps<T>) {
  
  // デバッグ用: currentStepの変更を監視
  useEffect(() => {
    if (debugMode) {
      console.log('StepNavigation: currentStep changed to:', currentStep);
    }
  }, [currentStep, debugMode]);

  const getStepStatus = (stepKey: T): StepStatus => {
    const currentIndex = steps.findIndex(s => s.key === currentStep);
    const stepIndex = steps.findIndex(s => s.key === stepKey);
    
    if (stepIndex < currentIndex) return 'completed';
    if (stepIndex === currentIndex) return 'current';
    return 'upcoming';
  };

  const handleStepClick = (stepKey: T) => {
    if (debugMode) {
      console.log('Step clicked:', stepKey);
      console.log('Current step before:', currentStep);
      console.log('Can navigate:', canNavigateToStep(stepKey));
    }
    
    if (canNavigateToStep(stepKey)) {
      if (debugMode) {
        console.log('Navigating to step:', stepKey);
      }
      
      try {
        onStepChange(stepKey);
        if (debugMode) {
          console.log('onStepChange called successfully');
        }
      } catch (error) {
        console.error('Error in onStepChange:', error);
      }
    } else {
      if (debugMode) {
        console.log('Navigation blocked for step:', stepKey);
      }
    }
  };

  const getCurrentStepInfo = () => {
    return steps.find(s => s.key === currentStep);
  };

  const getProgressPercentage = () => {
    const currentIndex = steps.findIndex(s => s.key === currentStep);
    return ((currentIndex + 1) / steps.length) * 100;
  };

  return (
    <div className={`mb-8 ${className}`}>
      {/* デスクトップ表示 */}
      <div className="hidden md:flex items-center justify-center">
        {steps.map((step, index) => {
          const Icon = step.icon;
          const status = getStepStatus(step.key);
          const canNavigate = canNavigateToStep(step.key);
          
          return (
            <div key={step.key} className="flex items-center">
              {/* ステップアイコンとラベル全体をクリック可能に */}
              <div className="flex flex-col items-center">
                <button
                  onClick={() => handleStepClick(step.key)}
                  disabled={!canNavigate}
                  className={`
                    flex items-center justify-center w-12 h-12 rounded-full border-2 transition-all duration-200
                    ${status === 'completed' 
                      ? 'bg-green-600 border-green-600 text-white hover:bg-green-700' 
                      : status === 'current'
                        ? 'bg-blue-600 border-blue-600 text-white'
                        : canNavigate
                          ? 'border-blue-300 text-blue-500 bg-white hover:border-blue-500 hover:bg-blue-50'
                          : 'border-gray-300 text-gray-400 bg-white'
                    }
                    ${canNavigate ? 'hover:scale-105 cursor-pointer' : 'cursor-not-allowed'}
                  `}
                  title={getNavigationMessage(step.key)}
                >
                  {status === 'completed' ? (
                    <CheckCircle className="h-6 w-6" />
                  ) : (
                    <Icon className="h-6 w-6" />
                  )}
                </button>
                
                {/* ステップ情報もクリック可能に */}
                <button
                  onClick={() => handleStepClick(step.key)}
                  disabled={!canNavigate}
                  className={`
                    mt-2 text-center transition-all duration-200 p-2 rounded-md
                    ${canNavigate ? 'hover:bg-gray-50 cursor-pointer' : 'cursor-not-allowed'}
                  `}
                  title={getNavigationMessage(step.key)}
                >
                  <div className={`text-sm font-medium ${
                    status === 'current' ? 'text-blue-600' : 
                    status === 'completed' ? 'text-green-600' : 
                    canNavigate ? 'text-gray-700 hover:text-blue-600' :
                    'text-gray-500'
                  }`}>
                    {step.label}
                  </div>
                  <div className={`text-xs mt-1 ${
                    status === 'current' ? 'text-blue-500' :
                    status === 'completed' ? 'text-green-500' :
                    canNavigate ? 'text-gray-600 hover:text-blue-500' :
                    'text-gray-500'
                  }`}>
                    {step.description}
                  </div>
                </button>
              </div>
              
              {/* 接続線 */}
              {index < steps.length - 1 && (
                <div className={`w-16 h-0.5 mx-4 transition-colors duration-200 ${
                  getStepStatus(steps[index + 1].key) === 'completed' || 
                  getStepStatus(steps[index + 1].key) === 'current'
                    ? 'bg-blue-300' 
                    : 'bg-gray-300'
                }`} />
              )}
            </div>
          );
        })}
      </div>

      {/* モバイル表示 */}
      <div className="md:hidden">
        <div className="flex items-center justify-between px-4">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const status = getStepStatus(step.key);
            const canNavigate = canNavigateToStep(step.key);
            
            return (
              <div key={step.key} className="flex flex-col items-center flex-1">
                <button
                  onClick={() => handleStepClick(step.key)}
                  disabled={!canNavigate}
                  className={`
                    flex items-center justify-center w-10 h-10 rounded-full border-2 transition-all duration-200 mb-2
                    ${status === 'completed' 
                      ? 'bg-green-600 border-green-600 text-white' 
                      : status === 'current'
                        ? 'bg-blue-600 border-blue-600 text-white'
                        : canNavigate
                          ? 'border-blue-300 text-blue-500 bg-white'
                          : 'border-gray-300 text-gray-400 bg-white'
                    }
                    ${canNavigate ? 'cursor-pointer' : 'cursor-not-allowed'}
                  `}
                  title={getNavigationMessage(step.key)}
                >
                  {status === 'completed' ? (
                    <CheckCircle className="h-5 w-5" />
                  ) : (
                    <Icon className="h-5 w-5" />
                  )}
                </button>
                
                <div className={`text-xs text-center font-medium ${
                  status === 'current' ? 'text-blue-600' : 
                  status === 'completed' ? 'text-green-600' : 
                  canNavigate ? 'text-gray-700' :
                  'text-gray-500'
                }`}>
                  {step.shortLabel}
                </div>
              </div>
            );
          })}
        </div>
        
        {/* モバイル用プログレスバー */}
        {showProgressBar && (
          <div className="mt-4 px-4">
            <div className="flex justify-between text-xs text-gray-500 mb-1">
              <span>進捗</span>
              <span>{steps.findIndex(s => s.key === currentStep) + 1} / {steps.length}</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${getProgressPercentage()}%` }}
              />
            </div>
          </div>
        )}
      </div>
      
      {/* 現在のステップの説明 */}
      {showCurrentStepInfo && (
        <div className="mt-4 text-center">
          <div className="inline-flex items-center px-4 py-2 bg-blue-50 rounded-lg">
            <Info className="h-4 w-4 text-blue-600 mr-2" />
            <div className="text-sm text-blue-700">
              現在のステップ: <span className="font-medium">{getCurrentStepInfo()?.label}</span>
            </div>
          </div>
          
          {/* デバッグ用の状態表示 */}
          {debugMode && (
            <div className="mt-2 text-xs text-gray-500">
              デバッグ: currentStep = "{currentStep}"
            </div>
          )}
        </div>
      )}
    </div>
  );
} 