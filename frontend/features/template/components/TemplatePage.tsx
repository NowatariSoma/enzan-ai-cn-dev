'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/layout/card';
import { Button } from '@/components/ui/forms/button';
import { Badge } from '@/components/ui/feedback/badge';
import { Layout, Palette, Smartphone, Monitor, Tablet, Code2 } from 'lucide-react';

export function TemplatePage() {
  return (
    <div className="space-y-8">
      {/* レイアウト概要 */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layout className="h-6 w-6 text-blue-600" />
            共通レイアウトテンプレート
          </CardTitle>
          <CardDescription>
            すべてのページで統一されたレイアウト構造を提供し、コンテンツの一貫性を保ちます
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center gap-3 p-4 bg-white rounded-lg border">
              <Monitor className="h-8 w-8 text-blue-600" />
              <div>
                <div className="font-semibold">デスクトップ対応</div>
                <div className="text-sm text-gray-600">固定サイドバー + ヘッダー</div>
              </div>
            </div>
            <div className="flex items-center gap-3 p-4 bg-white rounded-lg border">
              <Tablet className="h-8 w-8 text-green-600" />
              <div>
                <div className="font-semibold">タブレット対応</div>
                <div className="text-sm text-gray-600">レスポンシブレイアウト</div>
              </div>
            </div>
            <div className="flex items-center gap-3 p-4 bg-white rounded-lg border">
              <Smartphone className="h-8 w-8 text-purple-600" />
              <div>
                <div className="font-semibold">モバイル対応</div>
                <div className="text-sm text-gray-600">モバイルサイドバー</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* レイアウト機能 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-white border border-gray-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Palette className="h-5 w-5 text-orange-600" />
              レイアウト機能
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">統一されたヘッダー</span>
                <Badge variant="default">実装済み</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">レスポンシブサイドバー</span>
                <Badge variant="default">実装済み</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">コンテンツ幅の統一</span>
                <Badge variant="default">実装済み</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">モバイル対応</span>
                <Badge variant="default">実装済み</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white border border-gray-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Code2 className="h-5 w-5 text-green-600" />
              使用方法
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="p-3 bg-gray-50 rounded-lg">
                <code className="text-sm text-gray-800">
                  {'<AppTemplate title="ページタイトル" description="説明">'}
                </code>
              </div>
              <div className="text-sm text-gray-600">
                このテンプレートを使用することで、すべてのページで統一されたレイアウトが適用されます。
              </div>
              <Button variant="outline" className="w-full">
                コンポーネントドキュメント
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* サンプルコンテンツ */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {Array.from({ length: 6 }, (_, i) => (
          <Card key={i} className="bg-white border border-gray-200 hover:shadow-md transition-shadow">
            <CardHeader>
              <CardTitle>サンプルカード {i + 1}</CardTitle>
              <CardDescription>
                統一されたレイアウト内でのコンテンツ表示例
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="h-3 bg-gray-200 rounded animate-pulse"></div>
                <div className="h-3 bg-gray-200 rounded animate-pulse w-3/4"></div>
                <div className="h-3 bg-gray-200 rounded animate-pulse w-1/2"></div>
              </div>
              <Button variant="outline" size="sm" className="mt-4">
                詳細を見る
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* レスポンシブテスト */}
      <Card className="bg-white border border-gray-200">
        <CardHeader>
          <CardTitle>レスポンシブデザインテスト</CardTitle>
          <CardDescription>
            画面サイズを変更して、レイアウトの動作を確認してください
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            <div className="p-4 bg-red-50 rounded-lg text-center">
              <div className="text-sm font-semibold text-red-800">XS (&lt; 640px)</div>
              <div className="text-xs text-red-600 mt-1">モバイル</div>
            </div>
            <div className="p-4 bg-yellow-50 rounded-lg text-center">
              <div className="text-sm font-semibold text-yellow-800">SM (≥ 640px)</div>
              <div className="text-xs text-yellow-600 mt-1">小画面</div>
            </div>
            <div className="p-4 bg-green-50 rounded-lg text-center">
              <div className="text-sm font-semibold text-green-800">MD (≥ 768px)</div>
              <div className="text-xs text-green-600 mt-1">タブレット</div>
            </div>
            <div className="p-4 bg-blue-50 rounded-lg text-center">
              <div className="text-sm font-semibold text-blue-800">LG (≥ 1024px)</div>
              <div className="text-xs text-blue-600 mt-1">デスクトップ</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 