from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import List, Optional, Dict, Any
from ..models.location import Location, Feature, LocationFeature
from ..schemas.locations import LocationCreate, LocationUpdate


def get_location(db: Session, location_id: int) -> Optional[Location]:
    return db.query(Location).filter(Location.id == location_id).first()


def get_locations(db: Session, skip: int = 0, limit: int = 100) -> List[Location]:
    return db.query(Location).offset(skip).limit(limit).all()


def get_location_features(db: Session, location_id: int) -> List[LocationFeature]:
    return db.query(LocationFeature).filter(LocationFeature.location_id == location_id).all()


def get_all_features(db: Session) -> List[Feature]:
    return db.query(Feature).filter(Feature.is_active == True).all()


def get_available_features(db: Session, location_id: int) -> Dict[str, bool]:
    """拠点で利用可能な機能一覧を取得（フロントエンド形式）"""
    feature_mapping = {
        'ai_measurement': 'aiMeasurement',
        'data_analysis': 'measurement',
        'reporting': 'reportGeneration',
        'user_management': 'userManagement',
        'location_management': 'locationManagement',
        'custom': 'simulation',
    }
    
    # すべての機能をfalseで初期化
    features_dict = {}
    all_features = get_all_features(db)
    for feature in all_features:
        frontend_key = feature_mapping.get(feature.feature_type)
        if frontend_key:
            features_dict[frontend_key] = False
    
    # 拠点で有効な機能をtrueに設定
    location_features = get_location_features(db, location_id)
    for location_feature in location_features:
        if location_feature.is_enabled:
            feature = db.query(Feature).filter(Feature.id == location_feature.feature_id).first()
            if feature:
                frontend_key = feature_mapping.get(feature.feature_type)
                if frontend_key:
                    features_dict[frontend_key] = True
    
    return features_dict


def create_location(db: Session, location: LocationCreate) -> Location:
    db_location = Location(**location.dict())
    db.add(db_location)
    db.commit()
    db.refresh(db_location)
    return db_location


def update_location(db: Session, location_id: int, location: LocationUpdate) -> Optional[Location]:
    db_location = get_location(db, location_id)
    if db_location:
        update_data = location.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_location, field, value)
        db.commit()
        db.refresh(db_location)
    return db_location


def delete_location(db: Session, location_id: int) -> bool:
    db_location = get_location(db, location_id)
    if db_location:
        db.delete(db_location)
        db.commit()
        return True
    return False