# StepNavigation 共通コンポーネント

## 概要

`StepNavigation`は、ステップ形式のナビゲーションを表示するための汎用的なReactコンポーネントです。
4つのステップ（ヒアリング入力、AI分析、対話質問、プレビュー）のような、順序立てられた作業フローを視覚的に表現します。

## 特徴

- **汎用性**: 任意のステップ型に対応（TypeScriptジェネリクス使用）
- **レスポンシブ**: デスクトップとモバイル両方に対応
- **カスタマイズ可能**: ナビゲーション制御、メッセージ、表示オプションをカスタマイズ可能
- **アクセシビリティ**: キーボードナビゲーションとスクリーンリーダー対応

## 基本的な使用方法

```tsx
import { StepNavigation } from '@/shared/components';
import { StepDefinition } from '@/shared/types';
import { FileText, Search, MessageCircle, Eye } from 'lucide-react';

// ステップの型定義
type MyStep = 'input' | 'analysis' | 'dialogue' | 'preview';

// ステップ定義
const steps: StepDefinition<MyStep>[] = [
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
  // ... その他のステップ
];

// コンポーネント内での使用
function MyComponent() {
  const [currentStep, setCurrentStep] = useState<MyStep>('input');

  return (
    <StepNavigation
      steps={steps}
      currentStep={currentStep}
      onStepChange={setCurrentStep}
    />
  );
}
```

## Props

### 必須Props

| Prop | 型 | 説明 |
|------|-----|------|
| `steps` | `StepDefinition<T>[]` | ステップの定義配列 |
| `currentStep` | `T` | 現在のステップ |
| `onStepChange` | `(step: T) => void` | ステップ変更時のコールバック |

### オプションProps

| Prop | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `canNavigateToStep` | `(stepKey: T) => boolean` | `() => true` | ステップへの移動可否を判定 |
| `getNavigationMessage` | `(stepKey: T) => string` | 自動生成 | ツールチップメッセージを生成 |
| `className` | `string` | `''` | 追加のCSSクラス |
| `showProgressBar` | `boolean` | `true` | モバイル用プログレスバーの表示 |
| `showCurrentStepInfo` | `boolean` | `true` | 現在のステップ情報の表示 |
| `debugMode` | `boolean` | `false` | デバッグログの出力 |

## StepDefinition型

```tsx
interface StepDefinition<T extends string = string> {
  key: T;                    // ステップの一意キー
  label: string;             // ステップの表示名
  description: string;       // ステップの説明
  shortLabel: string;        // モバイル表示用の短縮名
  icon: LucideIcon;         // ステップのアイコン
}
```

## 高度な使用例

### ナビゲーション制御

```tsx
<StepNavigation
  steps={steps}
  currentStep={currentStep}
  onStepChange={setCurrentStep}
  canNavigateToStep={(stepKey) => {
    // 順番にアクセス可能な制限
    const currentIndex = steps.findIndex(s => s.key === currentStep);
    const targetIndex = steps.findIndex(s => s.key === stepKey);
    return targetIndex <= currentIndex + 1;
  }}
  getNavigationMessage={(stepKey) => {
    const step = steps.find(s => s.key === stepKey);
    return step ? `${step.label}に移動` : '移動できません';
  }}
/>
```

### カスタムスタイリング

```tsx
<StepNavigation
  steps={steps}
  currentStep={currentStep}
  onStepChange={setCurrentStep}
  className="my-custom-navigation"
  showProgressBar={false}
  showCurrentStepInfo={false}
/>
```

## ステップの状態

コンポーネントは自動的に以下の状態を管理します：

- **completed**: 現在のステップより前のステップ（緑色）
- **current**: 現在のステップ（青色）
- **upcoming**: 現在のステップより後のステップ（グレー）

## レスポンシブデザイン

- **デスクトップ**: 横並びのステップ表示、詳細な説明付き
- **モバイル**: コンパクトなアイコン表示、プログレスバー付き

## アクセシビリティ

- ボタンにはaria-labelとtitle属性を設定
- キーボードナビゲーション対応
- 適切なカラーコントラスト

## 使用例ファイル

詳細な使用例は `StepNavigation.example.tsx` を参照してください。

## 既存コンポーネントからの移行

既存の`features/hearing-assistant/components/StepNavigation.tsx`は、この共通コンポーネントを使用するようにリファクタリングされています。同様のパターンで他の機能でも使用できます。