from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel


class StreamLogBase(BaseModel):
    frame_id: Optional[int] = None
    timestamp: Optional[int] = None
    message: str
    log_type: str = "info"


class StreamLogCreate(StreamLogBase):
    pass


class StreamLog(StreamLogBase):
    id: int
    stream_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class StreamAlertBase(BaseModel):
    frame_id: int
    timestamp: Optional[int] = None
    description: str
    is_acknowledged: bool = False
    is_confirmed_alert: bool = True


class StreamAlertCreate(StreamAlertBase):
    pass


class StreamAlert(StreamAlertBase):
    id: int
    stream_id: int
    created_at: datetime
    acknowledged_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class StreamBase(BaseModel):
    stream_id: str
    name: str
    ivs_url: str
    status: str = "inactive"


class StreamCreate(StreamBase):
    pass


class StreamUpdate(BaseModel):
    name: Optional[str] = None
    ivs_url: Optional[str] = None
    status: Optional[str] = None
    alert_count: Optional[int] = None
    thumbnail_url: Optional[str] = None
    frames_s3_path: Optional[str] = None
    logs_s3_path: Optional[str] = None
    alerts_s3_path: Optional[str] = None
    vectorstore_s3_path: Optional[str] = None
    stream_metadata: Optional[Dict[str, Any]] = None
    processed_frames: Optional[int] = None
    total_frames: Optional[int] = None
    processing_progress: Optional[int] = None


class Stream(StreamBase):
    id: int
    alert_count: int = 0
    thumbnail_url: Optional[str] = None
    frames_s3_path: Optional[str] = None
    logs_s3_path: Optional[str] = None
    alerts_s3_path: Optional[str] = None
    vectorstore_s3_path: Optional[str] = None
    stream_metadata: Optional[Dict[str, Any]] = None
    processed_frames: Optional[int] = None
    total_frames: Optional[int] = None
    processing_progress: Optional[int] = 0
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None
    user_id: int

    class Config:
        from_attributes = True


class StreamDetail(Stream):
    logs: List[StreamLog] = []
    alerts: List[StreamAlert] = []

    class Config:
        from_attributes = True 