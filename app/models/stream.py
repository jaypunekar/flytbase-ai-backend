from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, DateTime, JSON, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..database import Base


class Stream(Base):
    __tablename__ = "streams"

    id = Column(Integer, primary_key=True, index=True)
    stream_id = Column(String, unique=True, index=True)  # Unique ID for the stream
    name = Column(String)  # Name to identify the stream
    ivs_url = Column(String)  # AWS IVS Playground URL
    status = Column(String, default="inactive")  # inactive, active, paused, error
    
    # Additional metadata
    alert_count = Column(Integer, default=0)
    thumbnail_url = Column(String, nullable=True)
    
    # Processing progress tracking
    processed_frames = Column(Integer, default=0)
    total_frames = Column(Integer, default=0)
    processing_progress = Column(Integer, default=0)  # Progress as percentage (0-100)
    
    # Storage paths
    frames_s3_path = Column(String, nullable=True)  # Directory containing extracted frames
    logs_s3_path = Column(String, nullable=True)  # Path to logs file
    alerts_s3_path = Column(String, nullable=True)  # Path to alerts file
    vectorstore_s3_path = Column(String, nullable=True)  # Path to vector store

    # JSON field for any extra metadata
    stream_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_active_at = Column(DateTime(timezone=True), nullable=True)
    
    # Foreign key to user
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    user = relationship("User", back_populates="streams")
    logs = relationship("StreamLog", back_populates="stream", cascade="all, delete-orphan")
    alerts = relationship("StreamAlert", back_populates="stream", cascade="all, delete-orphan")


class StreamLog(Base):
    __tablename__ = "stream_logs"

    id = Column(Integer, primary_key=True, index=True)
    frame_id = Column(Integer, nullable=True)
    timestamp = Column(Integer, nullable=True)
    message = Column(Text)
    log_type = Column(String, default="info")  # info, warning, error
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign key to stream
    stream_id = Column(Integer, ForeignKey("streams.id", ondelete="CASCADE"))
    
    # Relationship
    stream = relationship("Stream", back_populates="logs")


class StreamAlert(Base):
    __tablename__ = "stream_alerts"

    id = Column(Integer, primary_key=True, index=True)
    frame_id = Column(Integer)
    timestamp = Column(Integer, nullable=True)
    description = Column(Text)
    is_acknowledged = Column(Boolean, default=False)
    is_confirmed_alert = Column(Boolean, default=True)  # Indicates if this is a confirmed genuine alert
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    
    # Foreign key to stream
    stream_id = Column(Integer, ForeignKey("streams.id", ondelete="CASCADE"))
    
    # Relationship
    stream = relationship("Stream", back_populates="alerts") 