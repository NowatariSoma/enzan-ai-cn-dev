from app.api.endpoints import (
    analysis,
    cache_management,
    charts,
    csv_processing,
    displacement,
    displacement_analysis,
    locations,
    measurements,
    models,
    prediction,
    simulation,
)
from fastapi import APIRouter

api_router = APIRouter()

# 各エンドポイントのルーターを追加
api_router.include_router(displacement.router, prefix="/displacement", tags=["displacement"])

api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])

api_router.include_router(models.router, prefix="/models", tags=["models"])

api_router.include_router(simulation.router, prefix="/simulation", tags=["simulation"])

api_router.include_router(measurements.router, prefix="/measurements", tags=["measurements"])

api_router.include_router(
    prediction.router, prefix="/prediction", tags=["prediction", "machine-learning"]
)

api_router.include_router(csv_processing.router, prefix="/csv", tags=["csv-processing"])

api_router.include_router(charts.router, prefix="/charts", tags=["charts", "visualization"])

api_router.include_router(cache_management.router, prefix="/cache", tags=["cache-management"])

api_router.include_router(locations.router, tags=["locations"])

api_router.include_router(displacement_analysis.router, prefix="/displacement-analysis", tags=["displacement-analysis", "machine-learning"])
