"""
データフレームキャッシュ管理モジュール
API起動時にデータを読み込み、メモリ上にキャッシュして高速アクセスを実現
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from app.core.csv_loader import CSVDataLoader
from app.core.config import settings
import asyncio
from threading import Lock

logger = logging.getLogger(__name__)

class DataFrameCache:
    """データフレームをメモリ上にキャッシュするシングルトンクラス"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.cache: Dict[str, Dict[str, Any]] = {}
            self.csv_loader = CSVDataLoader()
            self.last_update: Dict[str, datetime] = {}
            self.initialized = True
    
    def load_folder_data(self, folder_name: str, max_distance_from_face: float = 100.0) -> bool:
        """
        指定フォルダのデータをキャッシュに読み込む
        
        Args:
            folder_name: データフォルダ名
            max_distance_from_face: 切羽からの最大距離
            
        Returns:
            bool: 読み込み成功の可否
        """
        try:
            logger.info(f"Loading data for folder: {folder_name}")
            
            # 入力フォルダのパス設定
            input_folder = settings.DATA_FOLDER / folder_name / "main_tunnel" / "CN_measurement_data"
            measurements_path = input_folder / "measurements_A"
            
            if not measurements_path.exists():
                logger.error(f"Measurements folder not found: {measurements_path}")
                return False
            
            measurement_a_csvs = list(measurements_path.glob("*.csv"))
            if not measurement_a_csvs:
                logger.error("No measurement CSV files found")
                return False
            
            # データフレーム生成
            df_all, dct_df_settlement, dct_df_convergence, dct_df_td, settlements, convergences = \
                self.csv_loader.generate_dataframes(measurement_a_csvs, max_distance_from_face)
            
            if df_all.empty:
                logger.error("No valid data found in measurement files")
                return False
            
            # キャッシュキーを生成（フォルダ名と最大距離の組み合わせ）
            cache_key = f"{folder_name}_{max_distance_from_face}"
            
            # キャッシュに保存
            self.cache[cache_key] = {
                'df_all': df_all,
                'dct_df_settlement': dct_df_settlement,
                'dct_df_convergence': dct_df_convergence,
                'dct_df_td': dct_df_td,
                'settlements': settlements,
                'convergences': convergences,
                'folder_name': folder_name,
                'max_distance_from_face': max_distance_from_face
            }
            
            self.last_update[cache_key] = datetime.now()
            logger.info(f"Successfully cached data for {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading folder data: {e}")
            return False
    
    def get_cached_data(self, folder_name: str, max_distance_from_face: float = 100.0) -> Optional[Dict[str, Any]]:
        """
        キャッシュからデータを取得。なければ自動で読み込み
        
        Args:
            folder_name: データフォルダ名
            max_distance_from_face: 切羽からの最大距離
            
        Returns:
            キャッシュされたデータ辞書、または None
        """
        cache_key = f"{folder_name}_{max_distance_from_face}"
        
        # キャッシュにない場合は読み込み
        if cache_key not in self.cache:
            success = self.load_folder_data(folder_name, max_distance_from_face)
            if not success:
                return None
        
        return self.cache.get(cache_key)
    
    def clear_cache(self, folder_name: Optional[str] = None):
        """
        キャッシュをクリア
        
        Args:
            folder_name: 特定フォルダのみクリアする場合に指定
        """
        if folder_name:
            # 特定フォルダのキャッシュをクリア
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(f"{folder_name}_")]
            for key in keys_to_remove:
                del self.cache[key]
                if key in self.last_update:
                    del self.last_update[key]
            logger.info(f"Cleared cache for folder: {folder_name}")
        else:
            # 全キャッシュをクリア
            self.cache.clear()
            self.last_update.clear()
            logger.info("Cleared all cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        キャッシュ情報を取得
        
        Returns:
            キャッシュの統計情報
        """
        return {
            'cached_folders': list(self.cache.keys()),
            'last_updates': {k: v.isoformat() for k, v in self.last_update.items()},
            'total_size': len(self.cache)
        }
    
    async def preload_all_folders(self):
        """
        利用可能な全フォルダのデータを事前読み込み（非同期）
        """
        try:
            data_folder = settings.DATA_FOLDER
            if not data_folder.exists():
                logger.warning(f"Data folder does not exist: {data_folder}")
                return
            
            # 全フォルダを検索
            folders = [f.name for f in data_folder.iterdir() if f.is_dir()]
            
            logger.info(f"Found {len(folders)} folders to preload")
            
            # 各フォルダのデータを読み込み
            for folder_name in folders:
                # デフォルトの距離設定で読み込み
                for distance in [50.0, 100.0, 200.0]:
                    self.load_folder_data(folder_name, distance)
                    await asyncio.sleep(0.1)  # 他のタスクにCPUを譲る
                    
        except Exception as e:
            logger.error(f"Error in preload_all_folders: {e}")

# シングルトンインスタンスを取得
def get_dataframe_cache() -> DataFrameCache:
    """DataFrameCacheのシングルトンインスタンスを取得"""
    return DataFrameCache()