from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, JSON, Numeric
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()


class Location(Base):
    __tablename__ = "locations_location"
    
    id = Column(Integer, primary_key=True, index=True)
    location_id = Column(String(100), nullable=True, unique=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    address = Column(Text, nullable=True)
    region = Column(String(100), nullable=True)
    prefecture = Column(String(100), nullable=True)
    tunnel_name = Column(String(255), nullable=True)
    folder_name = Column(String(255), nullable=True)
    status = Column(String(20), nullable=True, default='planning')
    start_date = Column(DateTime, nullable=True)
    total_length = Column(Float, nullable=True)
    progress = Column(Numeric(5, 2), nullable=True, default=0.0)
    measurement_count = Column(Integer, nullable=True, default=0)
    alert_level = Column(String(20), nullable=True, default='safe')
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @property
    def coordinates(self) -> Optional[Dict[str, float]]:
        if self.latitude is not None and self.longitude is not None:
            return {"lat": self.latitude, "lng": self.longitude}
        return None


class Feature(Base):
    __tablename__ = "locations_feature"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    feature_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class LocationFeature(Base):
    __tablename__ = "locations_locationfeature"
    
    id = Column(Integer, primary_key=True, index=True)
    location_id = Column(Integer, nullable=False)
    feature_id = Column(Integer, nullable=False)
    is_enabled = Column(Boolean, nullable=False, default=True)
    settings = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)