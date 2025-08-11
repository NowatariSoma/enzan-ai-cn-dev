"""
キャッシュ管理用エンドポイント
"""

import logging
from typing import Any, Dict

from app.core.dataframe_cache import get_dataframe_cache
from fastapi import APIRouter, HTTPException

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/cache/info")
async def get_cache_info() -> Dict[str, Any]:
    """
    現在のキャッシュ情報を取得
    """
    cache = get_dataframe_cache()
    return cache.get_cache_info()


@router.post("/cache/reload/{folder_name}")
async def reload_folder_cache(
    folder_name: str, max_distance_from_face: float = 100.0
) -> Dict[str, str]:
    """
    特定フォルダのキャッシュを再読み込み
    """
    cache = get_dataframe_cache()

    # 既存のキャッシュをクリア
    cache.clear_cache(folder_name)

    # 再読み込み
    success = cache.load_folder_data(folder_name, max_distance_from_face)

    if success:
        return {"status": "success", "message": f"Cache reloaded for {folder_name}"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to reload cache for {folder_name}")


@router.delete("/cache/clear")
async def clear_all_cache() -> Dict[str, str]:
    """
    全キャッシュをクリア
    """
    cache = get_dataframe_cache()
    cache.clear_cache()
    return {"status": "success", "message": "All cache cleared"}


@router.delete("/cache/clear/{folder_name}")
async def clear_folder_cache(folder_name: str) -> Dict[str, str]:
    """
    特定フォルダのキャッシュをクリア
    """
    cache = get_dataframe_cache()
    cache.clear_cache(folder_name)
    return {"status": "success", "message": f"Cache cleared for {folder_name}"}
