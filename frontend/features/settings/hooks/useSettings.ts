import { useState } from 'react';
import { Tab, UserProfile, TeamMember, NotificationSettings } from '../types';

const mockProfile: UserProfile = {
  name: 'ユーザー',
  email: 'user@example.com',
  role: 'admin',
  theme: 'light',
};

const mockTeamMembers: TeamMember[] = [
  {
    id: '1',
    name: 'ユーザー',
    email: 'user@example.com',
    role: 'admin',
    lastActive: '2024-01-15T10:30:00Z',
  },
  {
    id: '2',
    name: 'チームメンバー1',
    email: 'member1@example.com',
    role: 'user',
    lastActive: '2024-01-14T16:45:00Z',
  },
  {
    id: '3',
    name: 'チームメンバー2',
    email: 'member2@example.com',
    role: 'viewer',
    lastActive: '2024-01-13T09:15:00Z',
  },
];

const mockNotifications: NotificationSettings = {
  email: true,
  push: false,
  comments: true,
  uploads: true,
};

export const useSettings = () => {
  const [activeTab, setActiveTab] = useState<Tab>('profile');
  const [profile, setProfile] = useState<UserProfile>(mockProfile);
  const [teamMembers, setTeamMembers] = useState<TeamMember[]>(mockTeamMembers);
  const [notifications, setNotifications] = useState<NotificationSettings>(mockNotifications);

  const handleProfileUpdate = () => {
    // TODO: API呼び出し
    alert('プロフィールを更新しました');
  };

  const handleInviteMember = () => {
    const email = prompt('招待するメンバーのメールアドレスを入力してください');
    if (email) {
      // TODO: API呼び出し
      alert(`${email} に招待メールを送信しました`);
    }
  };

  const handleRoleChange = (memberId: string, newRole: 'admin' | 'user' | 'viewer') => {
    setTeamMembers(prev => 
      prev.map(member => 
        member.id === memberId ? { ...member, role: newRole } : member
      )
    );
  };

  const handleRemoveMember = (memberId: string) => {
    if (confirm('このメンバーを削除しますか？')) {
      setTeamMembers(prev => prev.filter(member => member.id !== memberId));
    }
  };

  const handleNotificationToggle = (key: keyof NotificationSettings) => {
    setNotifications(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  return {
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
  };
}; 