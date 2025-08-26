from pydantic import BaseModel, Field, RootModel
from datetime import datetime
from typing import Optional, Dict, Any, List


class LocationBase(BaseModel):
    location_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    address: Optional[str] = None
    region: Optional[str] = None
    prefecture: Optional[str] = None
    tunnel_name: Optional[str] = None
    folder_name: Optional[str] = None
    status: Optional[str] = "planning"
    start_date: Optional[datetime] = None
    total_length: Optional[float] = None
    progress: Optional[int] = 0
    measurement_count: Optional[int] = 0
    alert_level: Optional[str] = "safe"
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class LocationCreate(LocationBase):
    pass


class LocationUpdate(LocationBase):
    name: Optional[str] = None


class LocationResponse(LocationBase):
    id: int
    coordinates: Optional[Dict[str, float]] = None
    available_features: Dict[str, bool] = Field(default_factory=dict)
    lastUpdated: str
    user_count: int = 0
    feature_count: int = 0
    enabled_feature_count: int = 0
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


LocationListResponse = RootModel[List[LocationResponse]]