'use client';

import { Alert, AlertDescription } from '@/components/ui/feedback/alert';
import { AlertCircle } from 'lucide-react';

interface ErrorAlertProps {
  title: string;
  error?: string;
  className?: string;
}

export function ErrorAlert({ title, error, className }: ErrorAlertProps) {
  return (
    <Alert variant="destructive" className={className}>
      <AlertCircle className="h-4 w-4" />
      <AlertDescription>
        {title}
        {error && (
          <>
            <br />
            <span className="text-sm">エラー: {error}</span>
          </>
        )}
      </AlertDescription>
    </Alert>
  );
}