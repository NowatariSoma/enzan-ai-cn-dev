'use client';

import React from 'react';
import { AppTemplate } from '@/shared/components/layout/AppTemplate';
import { User, Users, Bell } from 'lucide-react';
import { useSettings } from '../hooks/useSettings';
import { ProfileTab } from './ProfileTab';
import { TeamTab } from './TeamTab';
import { NotificationsTab } from './NotificationsTab';

export const SettingsPage: React.FC = () => {
  const {
    activeTab,
    setActiveTab,
    profile,
    setProfile,
    teamMembers,
    notifications,
    handleProfileUpdate,
    handleInviteMember,
    handleRoleChange,
    handleRemoveMember,
    handleNotificationToggle,
  } = useSettings();

  const handleNotificationSave = () => {
    alert('通知設定を保存しました');
  };

  return (
    <AppTemplate
      title="設定"
      description="アカウント設定とチーム管理を行います"
      maxWidth="7xl"
    >
      {/* タブナビゲーション */}
      <div className="border-b border-gray-200 mb-8 bg-white">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('profile')}
            className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'profile'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <User className="w-4 h-4 inline mr-2" />
            プロフィール
          </button>
          <button
            onClick={() => setActiveTab('team')}
            className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'team'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <Users className="w-4 h-4 inline mr-2" />
            チーム管理
          </button>
          <button
            onClick={() => setActiveTab('notifications')}
            className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'notifications'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <Bell className="w-4 h-4 inline mr-2" />
            通知設定
          </button>
        </nav>
      </div>

      {/* タブコンテンツ */}
      <div className="min-h-[600px]">
        {activeTab === 'profile' && (
          <ProfileTab
            profile={profile}
            setProfile={setProfile}
            onUpdate={handleProfileUpdate}
          />
        )}

        {activeTab === 'team' && (
          <TeamTab
            teamMembers={teamMembers}
            onInviteMember={handleInviteMember}
            onRoleChange={handleRoleChange}
            onRemoveMember={handleRemoveMember}
          />
        )}

        {activeTab === 'notifications' && (
          <NotificationsTab
            notifications={notifications}
            onToggle={handleNotificationToggle}
            onSave={handleNotificationSave}
          />
        )}
      </div>
    </AppTemplate>
  );
}; 