import React from 'react';
import { Card, CardContent } from '@/components/ui/layout/card';
import { Button } from '@/components/ui/forms/button';
import { User, UserPlus } from 'lucide-react';
import { TeamMember } from '../types';
import { RoleBadge } from '../../../shared/components/RoleBadge';
import { formatDateTime } from '../../../shared/utils/format';

interface TeamTabProps {
  teamMembers: TeamMember[];
  onInviteMember: () => void;
  onRoleChange: (memberId: string, newRole: 'admin' | 'user' | 'viewer') => void;
  onRemoveMember: (memberId: string) => void;
}

export const TeamTab: React.FC<TeamTabProps> = ({
  teamMembers,
  onInviteMember,
  onRoleChange,
  onRemoveMember,
}) => {
  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">チームメンバー</h2>
          <p className="text-gray-600">チームメンバーの招待と権限管理</p>
        </div>
        <Button onClick={onInviteMember} className="bg-blue-600 text-white hover:bg-blue-700">
          <UserPlus className="w-4 h-4 mr-2" />
          メンバーを招待
        </Button>
      </div>

      <Card className="bg-white border border-gray-200">
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    メンバー
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    権限
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    最終アクティブ
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    操作
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {teamMembers.map((member) => (
                  <tr key={member.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="flex-shrink-0 h-10 w-10">
                          <div className="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center">
                            <User className="h-5 w-5 text-blue-600" />
                          </div>
                        </div>
                        <div className="ml-4">
                          <div className="text-sm font-medium text-gray-900">
                            {member.name}
                          </div>
                          <div className="text-sm text-gray-500">
                            {member.email}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <select
                        value={member.role}
                        onChange={(e) => onRoleChange(member.id, e.target.value as any)}
                        className="text-sm border border-gray-300 rounded px-2 py-1 bg-white"
                        disabled={member.id === '1'} // 自分の権限は変更不可
                      >
                        <option value="admin">管理者</option>
                        <option value="user">ユーザー</option>
                        <option value="viewer">閲覧者</option>
                      </select>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatDateTime(member.lastActive)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      {member.id !== '1' && ( // 自分は削除不可
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => onRemoveMember(member.id)}
                          className="text-red-600 hover:text-red-700 hover:bg-red-50"
                        >
                          削除
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}; 