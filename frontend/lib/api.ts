const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
const AUTH_URL = process.env.NEXT_PUBLIC_AUTH_URL;

// Debug: Log the URLs being used
console.log('API Configuration:', {
  API_BASE_URL,
  AUTH_URL,
  NODE_ENV: process.env.NODE_ENV,
});

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function fetchApi(url: string, options: RequestInit = {}) {
  const token = localStorage.getItem('authToken');
  
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` }),
      ...options.headers,
    },
  });

  if (!response.ok) {
    throw new ApiError(response.status, `HTTP error! status: ${response.status}`);
  }

  return response;
}

// Authentication
export const auth = {
  async login(username: string, password: string) {
    console.log('Login attempt:', { username, password: '***' });
    console.log('Auth URL:', `${AUTH_URL}/auth/login`);
    
    const response = await fetch(`${AUTH_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });

    console.log('Response status:', response.status);
    console.log('Response headers:', response.headers);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Login failed:', errorText);
      throw new ApiError(response.status, 'Invalid credentials');
    }

    const data = await response.json();
    console.log('Login successful:', data);
    
    // Save token to both localStorage and cookie
    localStorage.setItem('authToken', data.access_token);
    
    // Set cookie with secure options (session-only - expires when browser closes)
    document.cookie = `authToken=${data.access_token}; path=/; samesite=strict${
      window.location.protocol === 'https:' ? '; secure' : ''
    }`;
    
    console.log('Token saved to localStorage and session cookie');
    
    return data;
  },

  async logout() {
    console.log('Logging out...');
    localStorage.removeItem('authToken');
    
    // Remove cookie
    document.cookie = 'authToken=; path=/; expires=Thu, 01 Jan 1970 00:00:01 GMT;';
    
    console.log('Auth tokens cleared');
  },

  async getCurrentUser() {
    return fetchApi(`${AUTH_URL}/auth/me`);
  },

  // Debug function to clear all auth data
  clearAuthData() {
    console.log('Clearing all auth data...');
    localStorage.removeItem('authToken');
    localStorage.clear();
    document.cookie = 'authToken=; path=/; expires=Thu, 01 Jan 1970 00:00:01 GMT;';
    console.log('All auth data cleared');
  }
};

// Legacy API functions (keep for backward compatibility)
export async function fetchWorkerCounts() {
  const response = await fetchApi(`${API_BASE_URL}/api/workers/counts`);
  return response.json();
}

export async function fetchWorkerCountsFromAuth() {
  const response = await fetchApi(`${AUTH_URL}/api/workers/counts`);
  return response.json();
}

export async function fetchHistoricalData() {
  const response = await fetchApi(`${API_BASE_URL}/api/workers/historical`);
  return response.json();
}

export async function fetchHistoricalDataFromAuth() {
  const response = await fetchApi(`${AUTH_URL}/api/workers/historical`);
  return response.json();
}

export const historical = {
  async getData() {
    return fetchHistoricalData();
  },
  async getDataFromAuth() {
    return fetchHistoricalDataFromAuth();
  },
  async getHistoricalData(startDate: string, endDate: string) {
    const response = await fetchApi(`/api/db/historical?start=${startDate}&end=${endDate}`);
    return response.json();
  },
  async saveCurrentData() {
    // 現在のデータを履歴に保存（オプション）
    try {
      const response = await fetchApi(`${API_BASE_URL}/api/workers/save-current`);
      return response.json();
    } catch (error) {
      // 履歴保存は必須ではないので、エラーは無視
      console.warn('Failed to save current data to historical:', error);
      return null;
    }
  }
};

// Settings
export const settings = {
  async getProjectSettings() {
    return fetchApi(`${API_BASE_URL}/settings/project`);
  },

  async getUserSettings() {
    return fetchApi(`${API_BASE_URL}/settings/user`);
  }
};