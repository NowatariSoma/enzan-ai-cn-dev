'use client';

import { useState, useEffect } from 'react';
import { Location } from '@/lib/data/locations';

const STORAGE_KEY = 'favorite-locations';

export function useFavoriteLocations() {
  const [favoriteIds, setFavoriteIds] = useState<string[]>([]);

  // LocalStorageから読み込み
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          setFavoriteIds(parsed);
        }
      } catch (error) {
        console.error('Failed to parse favorite locations:', error);
      }
    }
  }, []);

  // LocalStorageに保存
  const saveFavorites = (ids: string[]) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(ids));
    setFavoriteIds(ids);
  };

  // お気に入りに追加
  const addFavorite = (locationId: string) => {
    if (!favoriteIds.includes(locationId)) {
      const newFavorites = [...favoriteIds, locationId];
      saveFavorites(newFavorites);
    }
  };

  // お気に入りから削除
  const removeFavorite = (locationId: string) => {
    const newFavorites = favoriteIds.filter(id => id !== locationId);
    saveFavorites(newFavorites);
  };

  // お気に入りのトグル
  const toggleFavorite = (locationId: string) => {
    if (favoriteIds.includes(locationId)) {
      removeFavorite(locationId);
    } else {
      addFavorite(locationId);
    }
  };

  // お気に入りかどうかチェック
  const isFavorite = (locationId: string) => {
    return favoriteIds.includes(locationId);
  };

  return {
    favoriteIds,
    addFavorite,
    removeFavorite,
    toggleFavorite,
    isFavorite
  };
}