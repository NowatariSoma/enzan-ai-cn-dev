import { LucideIcon } from 'lucide-react';

// 汎用的なステップ定義
export interface StepDefinition<T extends string = string> {
  key: T;
  label: string;
  description: string;
  shortLabel: string;
  icon: LucideIcon;
}

// ステップの状態
export type StepStatus = 'completed' | 'current' | 'upcoming';

// ナビゲーション可能性の判定関数
export type NavigationValidator<T extends string> = (stepKey: T) => boolean;

// ナビゲーションメッセージ生成関数
export type NavigationMessageGenerator<T extends string> = (stepKey: T) => string;

// 汎用ステップナビゲーションのプロパティ
export interface GenericStepNavigationProps<T extends string> {
  steps: StepDefinition<T>[];
  currentStep: T;
  onStepChange: (step: T) => void;
  canNavigateToStep?: NavigationValidator<T>;
  getNavigationMessage?: NavigationMessageGenerator<T>;
  className?: string;
  showProgressBar?: boolean;
  showCurrentStepInfo?: boolean;
  debugMode?: boolean;
} 