from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, DateTime, JSON, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..database import Base


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, unique=True, index=True)  # The unique video ID generated during upload
    filename = Column(String)
    s3_base_path = Column(String)  # Base S3 path where all video data is stored
    status = Column(String, default="uploaded")  # uploaded, processing, completed, error, interrupted
    
    # Additional metadata
    size_bytes = Column(Integer, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    resolution = Column(String, nullable=True)
    alert_count = Column(Integer, default=0)
    thumbnail_url = Column(String, nullable=True)
    
    # Processing progress tracking
    processed_frames = Column(Integer, default=0)
    total_frames = Column(Integer, default=0)
    processing_progress = Column(Integer, default=0)  # Progress as percentage (0-100)
    
    # Storage paths
    video_s3_path = Column(String, nullable=True)  # Original video path
    frames_s3_path = Column(String, nullable=True)  # Directory containing extracted frames
    logs_s3_path = Column(String, nullable=True)  # Path to logs file
    alerts_s3_path = Column(String, nullable=True)  # Path to alerts file
    vectorstore_s3_path = Column(String, nullable=True)  # Path to vector store

    # JSON field for any extra metadata
    video_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Foreign key to user
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    user = relationship("User", back_populates="videos")
    logs = relationship("VideoLog", back_populates="video", cascade="all, delete-orphan")
    alerts = relationship("VideoAlert", back_populates="video", cascade="all, delete-orphan")


class VideoLog(Base):
    __tablename__ = "video_logs"

    id = Column(Integer, primary_key=True, index=True)
    frame_id = Column(Integer, nullable=True)
    timestamp = Column(Integer, nullable=True)
    message = Column(Text)
    log_type = Column(String, default="info")  # info, warning, error
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign key to video
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"))
    
    # Relationship
    video = relationship("Video", back_populates="logs")


class VideoAlert(Base):
    __tablename__ = "video_alerts"

    id = Column(Integer, primary_key=True, index=True)
    frame_id = Column(Integer)
    timestamp = Column(Integer, nullable=True)
    description = Column(Text)
    is_acknowledged = Column(Boolean, default=False)
    is_confirmed_alert = Column(Boolean, default=True)  # Indicates if this is a confirmed genuine alert
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    
    # Foreign key to video
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"))
    
    # Relationship
    video = relationship("Video", back_populates="alerts") 