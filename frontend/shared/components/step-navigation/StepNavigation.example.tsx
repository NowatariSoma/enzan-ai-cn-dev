'use client';

import { useState } from 'react';
import { FileText, Search, MessageCircle, Eye, Settings, CheckSquare } from 'lucide-react';
import { StepNavigation } from './StepNavigation';
import { StepDefinition } from '../../types/navigation';

// 例1: ヒアリングアシスタント用のステップ
type HearingStep = 'input' | 'analysis' | 'dialogue' | 'preview';

const hearingSteps: StepDefinition<HearingStep>[] = [
  {
    key: 'input',
    label: 'ヒアリング入力',
    description: 'メモや議事録を入力',
    shortLabel: '入力',
    icon: FileText
  },
  {
    key: 'analysis',
    label: 'AI分析',
    description: 'キーワード抽出・マッピング',
    shortLabel: '分析',
    icon: Search
  },
  {
    key: 'dialogue',
    label: '対話質問',
    description: 'AIとの質疑応答',
    shortLabel: '対話',
    icon: MessageCircle
  },
  {
    key: 'preview',
    label: 'プレビュー',
    description: '要件定義書確認',
    shortLabel: '確認',
    icon: Eye
  }
];

// 例2: 一般的なプロジェクト設定用のステップ
type ProjectStep = 'setup' | 'configure' | 'review' | 'complete';

const projectSteps: StepDefinition<ProjectStep>[] = [
  {
    key: 'setup',
    label: 'プロジェクト設定',
    description: '基本情報の入力',
    shortLabel: '設定',
    icon: Settings
  },
  {
    key: 'configure',
    label: '詳細設定',
    description: '機能の詳細設定',
    shortLabel: '詳細',
    icon: Search
  },
  {
    key: 'review',
    label: 'レビュー',
    description: '設定内容の確認',
    shortLabel: '確認',
    icon: Eye
  },
  {
    key: 'complete',
    label: '完了',
    description: 'プロジェクトの作成',
    shortLabel: '完了',
    icon: CheckSquare
  }
];

export function StepNavigationExample() {
  const [hearingCurrentStep, setHearingCurrentStep] = useState<HearingStep>('input');
  const [projectCurrentStep, setProjectCurrentStep] = useState<ProjectStep>('setup');

  return (
    <div className="p-8 space-y-12">
      <div>
        <h2 className="text-2xl font-bold mb-6">ヒアリングアシスタント用ステップナビゲーション</h2>
        <StepNavigation
          steps={hearingSteps}
          currentStep={hearingCurrentStep}
          onStepChange={setHearingCurrentStep}
          canNavigateToStep={() => true}
          showProgressBar={true}
          showCurrentStepInfo={true}
          debugMode={true}
        />
        
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-600">
            現在選択されているステップ: <strong>{hearingCurrentStep}</strong>
          </p>
        </div>
      </div>

      <div>
        <h2 className="text-2xl font-bold mb-6">プロジェクト設定用ステップナビゲーション</h2>
        <StepNavigation
          steps={projectSteps}
          currentStep={projectCurrentStep}
          onStepChange={setProjectCurrentStep}
          canNavigateToStep={(stepKey) => {
            // 順番にアクセス可能な制限を設ける例
            const currentIndex = projectSteps.findIndex(s => s.key === projectCurrentStep);
            const targetIndex = projectSteps.findIndex(s => s.key === stepKey);
            return targetIndex <= currentIndex + 1;
          }}
          getNavigationMessage={(stepKey) => {
            const step = projectSteps.find(s => s.key === stepKey);
            return step ? `${step.label}に移動` : '移動できません';
          }}
          showProgressBar={true}
          showCurrentStepInfo={true}
          debugMode={false}
        />
        
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-600">
            現在選択されているステップ: <strong>{projectCurrentStep}</strong>
          </p>
          <p className="text-xs text-gray-500 mt-1">
            このナビゲーションでは、順番にアクセス制限が設定されています
          </p>
        </div>
      </div>
    </div>
  );
} 