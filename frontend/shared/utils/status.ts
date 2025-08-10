import { Status } from '../types/common';
import { CheckCircle, Clock, AlertCircle } from 'lucide-react';

export const getStatusIcon = (status: Status) => {
  switch (status) {
    case 'planning':
      return Clock;
    case 'in-progress':
      return AlertCircle;
    case 'completed':
      return CheckCircle;
    case 'on-hold':
      return AlertCircle;
    default:
      return Clock;
  }
};

export const getStatusColor = (status: Status): string => {
  switch (status) {
    case 'planning':
      return 'bg-yellow-100 text-yellow-800';
    case 'in-progress':
      return 'bg-blue-100 text-blue-800';
    case 'completed':
      return 'bg-green-100 text-green-800';
    case 'on-hold':
      return 'bg-red-100 text-red-800';
    default:
      return 'bg-gray-100 text-gray-800';
  }
};

export const getStatusText = (status: Status): string => {
  switch (status) {
    case 'planning':
      return '計画中';
    case 'in-progress':
      return '進行中';
    case 'completed':
      return '完了';
    case 'on-hold':
      return '保留';
    default:
      return status;
  }
};

export const getStatusIconColor = (status: Status): string => {
  switch (status) {
    case 'planning':
      return 'text-yellow-600';
    case 'in-progress':
      return 'text-blue-600';
    case 'completed':
      return 'text-green-600';
    case 'on-hold':
      return 'text-red-600';
    default:
      return 'text-gray-600';
  }
}; 