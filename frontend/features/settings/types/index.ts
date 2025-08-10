export type Tab = 'profile' | 'team' | 'notifications';

export interface UserProfile {
  name: string;
  email: string;
  role: string;
  theme: string;
}

export interface TeamMember {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'user' | 'viewer';
  lastActive: string;
}

export interface NotificationSettings {
  email: boolean;
  push: boolean;
  comments: boolean;
  uploads: boolean;
} 