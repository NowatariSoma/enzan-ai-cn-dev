'use client';

import { useRouter } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/layout/card';
import { Button } from '@/components/ui/forms/button';
import { FileText } from 'lucide-react';

export function RecentProjects() {
  const router = useRouter();

  return (
    <Card className="bg-white border border-gray-200">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileText className="h-5 w-5" />
          最近のプロジェクト
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-center text-gray-500 py-8">
          まだプロジェクトが作成されていません
          <div className="mt-2">
            <Button 
              variant="outline" 
              onClick={() => router.push('/simulation')}
              className="mt-2"
            >
              シミュレーション分析を開始
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 