'use client';

import { useState, useEffect } from 'react';
import { locationsAPI, type Location } from '@/lib/api/locations';

export function useLocations() {
  const [locations, setLocations] = useState<Location[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchLocations = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await locationsAPI.getLocations();
        setLocations(data);
      } catch (err) {
        console.error('Error fetching locations:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch locations');
        
        // APIエラーの場合はモックデータにフォールバック
        try {
          const { locations: mockLocations } = await import('@/lib/data/locations');
          setLocations(mockLocations);
        } catch (mockErr) {
          console.error('Failed to load mock data as fallback:', mockErr);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchLocations();
  }, []);

  const refetch = async () => {
    const fetchLocations = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await locationsAPI.getLocations();
        setLocations(data);
      } catch (err) {
        console.error('Error fetching locations:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch locations');
      } finally {
        setLoading(false);
      }
    };

    await fetchLocations();
  };

  return {
    locations,
    loading,
    error,
    refetch
  };
}

export function useLocation(locationId: string) {
  const [location, setLocation] = useState<Location | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!locationId) {
      setLocation(null);
      setLoading(false);
      return;
    }

    const fetchLocation = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await locationsAPI.getLocation(locationId);
        setLocation(data);
      } catch (err) {
        console.error('Error fetching location:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch location');
        
        // APIエラーの場合はモックデータにフォールバック
        try {
          const { getLocationById, locations: mockLocations } = await import('@/lib/data/locations');
          const mockLocation = getLocationById(locationId);
          setLocation(mockLocation || null);
        } catch (mockErr) {
          console.error('Failed to load mock data as fallback:', mockErr);
          setLocation(null);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchLocation();
  }, [locationId]);

  return {
    location,
    loading,
    error
  };
}