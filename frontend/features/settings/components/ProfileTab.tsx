import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/layout/card';
import { Button } from '@/components/ui/forms/button';
import { Shield, Sun, Moon, Monitor } from 'lucide-react';
import { UserProfile } from '../types';
import { RoleBadge } from '../../../shared/components/RoleBadge';

interface ProfileTabProps {
  profile: UserProfile;
  setProfile: React.Dispatch<React.SetStateAction<UserProfile>>;
  onUpdate: () => void;
}

export const ProfileTab: React.FC<ProfileTabProps> = ({ profile, setProfile, onUpdate }) => {
  return (
    <div className="max-w-2xl">
      <Card className="bg-white border border-gray-200">
        <CardHeader>
          <CardTitle>プロフィール情報</CardTitle>
          <CardDescription>
            アカウントの基本情報を管理します
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                名前
              </label>
              <input
                type="text"
                value={profile.name}
                onChange={(e) => setProfile(prev => ({ ...prev, name: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                メールアドレス
              </label>
              <input
                type="email"
                value={profile.email}
                onChange={(e) => setProfile(prev => ({ ...prev, email: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              権限
            </label>
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-gray-500" />
              <RoleBadge role={profile.role as any} />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              テーマ
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => setProfile(prev => ({ ...prev, theme: 'light' }))}
                className={`flex items-center gap-2 px-3 py-2 border rounded-md transition-colors ${
                  profile.theme === 'light'
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-300 hover:bg-gray-50'
                }`}
              >
                <Sun className="w-4 h-4" />
                ライト
              </button>
              <button
                onClick={() => setProfile(prev => ({ ...prev, theme: 'dark' }))}
                className={`flex items-center gap-2 px-3 py-2 border rounded-md transition-colors ${
                  profile.theme === 'dark'
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-300 hover:bg-gray-50'
                }`}
              >
                <Moon className="w-4 h-4" />
                ダーク
              </button>
              <button
                onClick={() => setProfile(prev => ({ ...prev, theme: 'system' }))}
                className={`flex items-center gap-2 px-3 py-2 border rounded-md transition-colors ${
                  profile.theme === 'system'
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-300 hover:bg-gray-50'
                }`}
              >
                <Monitor className="w-4 h-4" />
                システム
              </button>
            </div>
          </div>

          <div className="flex justify-end">
            <Button onClick={onUpdate} className="bg-blue-600 text-white hover:bg-blue-700">
              更新
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}; 