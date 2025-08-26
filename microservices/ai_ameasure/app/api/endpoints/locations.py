from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from ...db.database import get_db
from ...crud import locations as crud_locations
from ...schemas.locations import LocationResponse, LocationCreate, LocationUpdate
from ...models.location import Location

router = APIRouter()


@router.get("/locations", response_model=List[LocationResponse])
def get_locations(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """拠点一覧取得"""
    locations = crud_locations.get_locations(db, skip=skip, limit=limit)
    
    # フロントエンド形式でレスポンスを構築
    response_data = []
    for location in locations:
        available_features = crud_locations.get_available_features(db, location.id)
        
        location_data = {
            "id": location.id,
            "location_id": location.location_id,
            "name": location.name,
            "description": location.description,
            "address": location.address,
            "region": location.region,
            "prefecture": location.prefecture,
            "tunnel_name": location.tunnel_name,
            "folder_name": location.folder_name,
            "status": location.status,
            "start_date": location.start_date,
            "total_length": location.total_length,
            "progress": location.progress,
            "measurement_count": location.measurement_count,
            "alert_level": location.alert_level,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "coordinates": location.coordinates,
            "available_features": available_features,
            "lastUpdated": location.updated_at.isoformat(),
            "user_count": 0,  # TODO: 実装が必要
            "feature_count": len(available_features),
            "enabled_feature_count": sum(1 for enabled in available_features.values() if enabled),
            "created_at": location.created_at,
            "updated_at": location.updated_at,
        }
        response_data.append(location_data)
    
    return response_data


@router.get("/locations/{location_id}", response_model=LocationResponse)
def get_location(
    location_id: int,
    db: Session = Depends(get_db)
):
    """拠点詳細取得"""
    location = crud_locations.get_location(db, location_id=location_id)
    if location is None:
        raise HTTPException(status_code=404, detail="Location not found")
    
    available_features = crud_locations.get_available_features(db, location.id)
    
    location_data = {
        "id": location.id,
        "location_id": location.location_id,
        "name": location.name,
        "description": location.description,
        "address": location.address,
        "region": location.region,
        "prefecture": location.prefecture,
        "tunnel_name": location.tunnel_name,
        "folder_name": location.folder_name,
        "status": location.status,
        "start_date": location.start_date,
        "total_length": location.total_length,
        "progress": location.progress,
        "measurement_count": location.measurement_count,
        "alert_level": location.alert_level,
        "latitude": location.latitude,
        "longitude": location.longitude,
        "coordinates": location.coordinates,
        "available_features": available_features,
        "lastUpdated": location.updated_at.isoformat(),
        "user_count": 0,  # TODO: 実装が必要
        "feature_count": len(available_features),
        "enabled_feature_count": sum(1 for enabled in available_features.values() if enabled),
        "created_at": location.created_at,
        "updated_at": location.updated_at,
    }
    
    return location_data


@router.post("/locations", response_model=LocationResponse)
def create_location(
    location: LocationCreate,
    db: Session = Depends(get_db)
):
    """新規拠点作成"""
    db_location = crud_locations.create_location(db=db, location=location)
    available_features = crud_locations.get_available_features(db, db_location.id)
    
    location_data = {
        "id": db_location.id,
        "location_id": db_location.location_id,
        "name": db_location.name,
        "description": db_location.description,
        "address": db_location.address,
        "region": db_location.region,
        "prefecture": db_location.prefecture,
        "tunnel_name": db_location.tunnel_name,
        "folder_name": db_location.folder_name,
        "status": db_location.status,
        "start_date": db_location.start_date,
        "total_length": db_location.total_length,
        "progress": db_location.progress,
        "measurement_count": db_location.measurement_count,
        "alert_level": db_location.alert_level,
        "latitude": db_location.latitude,
        "longitude": db_location.longitude,
        "coordinates": db_location.coordinates,
        "available_features": available_features,
        "lastUpdated": db_location.updated_at.isoformat(),
        "user_count": 0,
        "feature_count": len(available_features),
        "enabled_feature_count": sum(1 for enabled in available_features.values() if enabled),
        "created_at": db_location.created_at,
        "updated_at": db_location.updated_at,
    }
    
    return location_data


@router.put("/locations/{location_id}", response_model=LocationResponse)
def update_location(
    location_id: int,
    location: LocationUpdate,
    db: Session = Depends(get_db)
):
    """拠点情報更新"""
    db_location = crud_locations.update_location(db=db, location_id=location_id, location=location)
    if db_location is None:
        raise HTTPException(status_code=404, detail="Location not found")
    
    available_features = crud_locations.get_available_features(db, db_location.id)
    
    location_data = {
        "id": db_location.id,
        "location_id": db_location.location_id,
        "name": db_location.name,
        "description": db_location.description,
        "address": db_location.address,
        "region": db_location.region,
        "prefecture": db_location.prefecture,
        "tunnel_name": db_location.tunnel_name,
        "folder_name": db_location.folder_name,
        "status": db_location.status,
        "start_date": db_location.start_date,
        "total_length": db_location.total_length,
        "progress": db_location.progress,
        "measurement_count": db_location.measurement_count,
        "alert_level": db_location.alert_level,
        "latitude": db_location.latitude,
        "longitude": db_location.longitude,
        "coordinates": db_location.coordinates,
        "available_features": available_features,
        "lastUpdated": db_location.updated_at.isoformat(),
        "user_count": 0,
        "feature_count": len(available_features),
        "enabled_feature_count": sum(1 for enabled in available_features.values() if enabled),
        "created_at": db_location.created_at,
        "updated_at": db_location.updated_at,
    }
    
    return location_data


@router.delete("/locations/{location_id}")
def delete_location(
    location_id: int,
    db: Session = Depends(get_db)
):
    """拠点削除"""
    success = crud_locations.delete_location(db=db, location_id=location_id)
    if not success:
        raise HTTPException(status_code=404, detail="Location not found")
    return {"message": "Location deleted successfully"}