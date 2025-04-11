from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel


class VideoLogBase(BaseModel):
    frame_id: Optional[int] = None
    timestamp: Optional[int] = None
    message: str
    log_type: str = "info"


class VideoLogCreate(VideoLogBase):
    pass


class VideoLog(VideoLogBase):
    id: int
    video_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class VideoAlertBase(BaseModel):
    frame_id: int
    timestamp: Optional[int] = None
    description: str
    is_acknowledged: bool = False
    is_confirmed_alert: bool = True


class VideoAlertCreate(VideoAlertBase):
    pass


class VideoAlert(VideoAlertBase):
    id: int
    video_id: int
    created_at: datetime
    acknowledged_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class VideoBase(BaseModel):
    video_id: str
    filename: str
    status: str = "uploaded"


class VideoCreate(VideoBase):
    s3_base_path: Optional[str] = None
    video_s3_path: Optional[str] = None


class VideoUpdate(BaseModel):
    status: Optional[str] = None
    alert_count: Optional[int] = None
    thumbnail_url: Optional[str] = None
    frames_s3_path: Optional[str] = None
    logs_s3_path: Optional[str] = None
    alerts_s3_path: Optional[str] = None
    vectorstore_s3_path: Optional[str] = None
    size_bytes: Optional[int] = None
    duration_seconds: Optional[int] = None
    resolution: Optional[str] = None
    video_metadata: Optional[Dict[str, Any]] = None
    processed_frames: Optional[int] = None
    total_frames: Optional[int] = None
    processing_progress: Optional[int] = None


class Video(VideoBase):
    id: int
    s3_base_path: Optional[str] = None
    size_bytes: Optional[int] = None
    duration_seconds: Optional[int] = None
    resolution: Optional[str] = None
    alert_count: int = 0
    thumbnail_url: Optional[str] = None
    video_s3_path: Optional[str] = None
    frames_s3_path: Optional[str] = None
    logs_s3_path: Optional[str] = None
    alerts_s3_path: Optional[str] = None
    vectorstore_s3_path: Optional[str] = None
    video_metadata: Optional[Dict[str, Any]] = None
    processed_frames: Optional[int] = None
    total_frames: Optional[int] = None
    processing_progress: Optional[int] = 0
    created_at: datetime
    updated_at: Optional[datetime] = None
    user_id: int

    class Config:
        from_attributes = True


class VideoDetail(Video):
    logs: List[VideoLog] = []
    alerts: List[VideoAlert] = []
    video_url: Optional[str] = None

    class Config:
        from_attributes = True 