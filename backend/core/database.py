"""
FlowVision Backend - Database Setup
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.core.config import DATABASE_URL, DATABASE_DIR
import os

# Ensure database directory exists
os.makedirs(DATABASE_DIR, exist_ok=True)

# Create engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class
Base = declarative_base()

# Database models
class FlowReading(Base):
    __tablename__ = "flow_readings"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True)
    flow_rate = Column(Float)
    pressure = Column(Float)
    temperature = Column(Float)
    is_leak = Column(Boolean, default=False)

class LeakDetection(Base):
    __tablename__ = "leak_detections"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True)
    leak_probability = Column(Float)
    is_leak_detected = Column(Boolean)
    detection_method = Column(String)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("[OK] Database initialized")

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
