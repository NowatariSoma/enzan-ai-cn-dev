import asyncio
import logging
from contextlib import asynccontextmanager

from app.api.api import api_router
from app.core.config import settings
from app.core.dataframe_cache import get_dataframe_cache
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動時の処理
    logger.info("Starting up - initializing dataframe cache...")
    cache = get_dataframe_cache()

    # バックグラウンドでデータを事前読み込み（非ブロッキング）
    asyncio.create_task(cache.preload_all_folders())
    logger.info("Dataframe cache initialization started in background")

    yield

    # 終了時の処理
    logger.info("Shutting down - clearing dataframe cache...")
    cache.clear_cache()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# APIルーターの追加
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
def read_root():
    return {
        "message": "Welcome to AI A-Measure API",
        "version": settings.VERSION,
        "docs": f"{settings.API_V1_STR}/docs",
    }
