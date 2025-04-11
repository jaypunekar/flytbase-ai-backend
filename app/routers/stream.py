import os
import uuid
import shutil
import tempfile
import json
import time
import boto3
import cv2
import base64
import threading
from fastapi import APIRouter, Depends, HTTPException, status, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from collections import defaultdict
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

# LangChain imports
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import openai

from ..auth import get_current_active_user, get_current_verified_user
from ..models.user import User
from ..models.stream import Stream, StreamLog, StreamAlert
from ..schemas.stream import StreamCreate, StreamUpdate, Stream as StreamSchema, StreamDetail, StreamAlert as StreamAlertSchema
from ..database import get_db
from ..utils.email import send_alert_email
from ..routers.video import is_genuinely_suspicious, detect_objects, encode_image, upload_to_s3, get_s3_url

router = APIRouter(
    prefix="/stream",
    tags=["stream"],
)

# Global variables for streaming sessions
streaming_sessions = {}

class StreamSession:
    def __init__(self, user_id, stream_id, ivs_url):
        self.user_id = user_id
        self.stream_id = stream_id
        self.ivs_url = ivs_url
        self.frame_knowledge = []
        self.current_vectorstore = None
        self.alerted_descriptions = set()
        self.object_memory = defaultdict(list)
        self.processing = False
        self.progress = 0
        self.total_frames = 0
        self.processed_frames = 0
        self.temp_dir = tempfile.mkdtemp()
        self.last_detection = ""
        self.static_frame_count = 0
        
    def cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def log_stream_alert(frame_id, description, suspicious_reason, user_id, stream_session, db_session=None, db_stream=None, user_email=None, timestamp=None):
    """
    Log a suspicious activity alert for a stream
    
    Args:
        frame_id: The frame ID where the suspicious activity was detected
        description: The description of what was detected in the frame
        suspicious_reason: The specific reason why this activity is considered suspicious
        user_id: User ID
        stream_session: StreamSession object
        db_session: Database session for retrieving user info
        db_stream: Stream database object
        user_email: User's email address
        timestamp: Timestamp of the frame
    """
    alert_key = f"{frame_id}_{suspicious_reason[:20]}"  # Create a key to avoid duplicate alerts
    
    if alert_key not in stream_session.alerted_descriptions:
        alert_file = os.path.join(stream_session.temp_dir, f"{stream_session.stream_id}_alerts.txt")
        
        # Extract confidence information from the suspicious reason
        confidence_level = "HIGH" if any(keyword in suspicious_reason.lower() for keyword in [
            "weapon", "gun", "knife", "assault", "breaking", "forced entry", "theft"
        ]) else "MEDIUM"
        
        # Log the alert
        alert_message = f"[ALERT-{confidence_level}] Frame {frame_id}: {suspicious_reason}\nDetails: {description}"
        with open(alert_file, "a") as f:
            f.write(alert_message + "\n\n")
            
        alert_s3_key = upload_to_s3(alert_file, f"logs/alerts.txt", user_id, stream_session.stream_id)
        stream_session.alerted_descriptions.add(alert_key)
        
        # Always try to send an email alert if we have user email
        try:
            if user_email and db_stream:
                # Try to get thumbnail URL for the frame
                frame_thumbnail_url = None
                try:
                    frame_s3_path = f"users/{user_id}/{stream_session.stream_id}/frames/frame_{frame_id}.jpg"
                    frame_thumbnail_url = get_s3_url(frame_s3_path)
                except Exception as e:
                    print(f"Could not get thumbnail URL: {str(e)}")
                    
                # Log that we're attempting to send an email alert
                print(f"Sending stream alert email to {user_email} for suspicious activity in stream {db_stream.name}")
                
                # Send email alert
                email_sent = send_alert_email(
                    user_email, 
                    f"Stream: {db_stream.name}", 
                    description, 
                    frame_id, 
                    timestamp or 0, 
                    frame_thumbnail_url
                )
                
                # Log whether the email was sent successfully
                if email_sent:
                    print(f"Stream alert email sent successfully to {user_email}")
                    with open(alert_file, "a") as f:
                        f.write(f"[EMAIL ALERT SENT] Email notification sent to {user_email}\n\n")
                else:
                    print(f"Failed to send stream alert email to {user_email}")
                    with open(alert_file, "a") as f:
                        f.write(f"[EMAIL ALERT FAILED] Could not send email notification to {user_email}\n\n")
        except Exception as e:
            print(f"Exception when sending stream alert email: {str(e)}")
            with open(alert_file, "a") as f:
                f.write(f"[EMAIL ALERT ERROR] {str(e)}\n\n")


def process_stream(stream_id, user_id, db_session=None):
    """
    Process frames from a live stream
    
    Args:
        stream_id: ID of the stream to process
        user_id: User ID
        db_session: Database session
    """
    if user_id not in streaming_sessions or stream_id not in streaming_sessions[user_id]:
        print(f"Stream session not found: user_id={user_id}, stream_id={stream_id}")
        return False
    
    session = streaming_sessions[user_id][stream_id]
    session.processing = True
    
    # Update stream status in DB
    db_stream = None
    if db_session:
        db_stream = db_session.query(Stream).filter(Stream.stream_id == stream_id).first()
        if db_stream:
            db_stream.status = "active"
            db_stream.last_active_at = datetime.now()
            db_session.commit()
    
    # Get user email for alerts
    user_email = None
    if db_session:
        user = db_session.query(User).filter(User.id == user_id).first()
        if user:
            user_email = user.email
            print(f"Retrieved user email for stream alerts: {user_email}")
    
    # Create directories
    frames_dir = os.path.join(session.temp_dir, f"{stream_id}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    log_file = os.path.join(session.temp_dir, f"{stream_id}_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"Starting stream processing for {stream_id}...\n")
    
    logs = []
    
    # Activity tracking for context awareness
    tracked_entities = {}  # Format: {entity_id: {"frames": [frame_ids], "descriptions": [descriptions]}}
    
    # Initialize embedding model
    embedding = OpenAIEmbeddings()
    language_model = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    # Custom prompt for answering questions about the stream
    custom_prompt = PromptTemplate.from_template(
        """You are analyzing a live video stream by reviewing image frames extracted from it. 
Each frame was analyzed to detect objects like humans, cars, their colors, and activities. Use all frame descriptions 
to answer questions about what's happening in the stream. You can summarize patterns, behavior, and suspicious activity.

Context:
{context}

Question: {question}
Answer:"""
    )
    
    # Alert tracking
    alert_count = 0
    confirmed_alerts = []
    
    # Capture from stream
    cap = cv2.VideoCapture(session.ivs_url)
    if not cap.isOpened():
        with open(log_file, 'a') as f:
            f.write(f"Error: Could not open stream URL: {session.ivs_url}\n")
        print(f"Error: Could not open stream URL: {session.ivs_url}")
        
        if db_session and db_stream:
            db_stream.status = "error"
            db_session.commit()
        
        session.processing = False
        return False
    
    frame_id = 0
    FRAME_INTERVAL = 30  # Process every 30th frame (adjust based on stream FPS)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if can't determine
    
    # Periodically update database with progress
    last_db_update_time = time.time()
    db_update_interval = 15  # seconds - more frequent for live streams
    
    frames_s3_path = None
    thumbnail_path = None
    
    # Main processing loop
    try:
        while cap.isOpened() and session.processing:
            ret, frame = cap.read()
            if not ret:
                # If the stream ended, try to reconnect
                with open(log_file, 'a') as f:
                    f.write(f"Stream connection lost, attempting to reconnect...\n")
                print(f"Stream connection lost, attempting to reconnect...")
                
                # Wait a bit before reconnecting
                time.sleep(2)
                cap.release()
                cap = cv2.VideoCapture(session.ivs_url)
                if not cap.isOpened():
                    with open(log_file, 'a') as f:
                        f.write(f"Failed to reconnect to stream\n")
                    break
                continue
            
            # Update progress tracking
            session.processed_frames = frame_id
            current_time = time.time()
            
            # Update DB periodically
            if db_session and db_stream and (current_time - last_db_update_time) > db_update_interval:
                db_stream.processed_frames = frame_id
                db_stream.last_active_at = datetime.now()
                db_session.commit()
                last_db_update_time = current_time
            
            # Process frame at regular intervals
            if frame_id % FRAME_INTERVAL == 0:
                timestamp = int(frame_id / fps)
                frame_path = os.path.join(frames_dir, f"frame_{frame_id}.jpg")
                cv2.imwrite(frame_path, frame)
                
                # Save first frame as thumbnail
                if frame_id == 0:
                    thumbnail_path = frame_path
                    # Upload thumbnail to S3
                    thumbnail_s3_key = upload_to_s3(thumbnail_path, "thumbnail.jpg", user_id, stream_id)
                    if thumbnail_s3_key and db_stream:
                        db_stream.thumbnail_url = get_s3_url(thumbnail_s3_key)
                        db_session.commit()
                
                # Log frame processing
                with open(log_file, 'a') as f:
                    f.write(f"Processing frame {frame_id} (Time: {timestamp}s)...\n")
                
                try:
                    # Upload frame to S3
                    frame_s3_key = upload_to_s3(frame_path, f"frames/frame_{frame_id}.jpg", user_id, stream_id)
                    if frame_id == 0 and frame_s3_key:
                        frames_s3_path = f"users/{user_id}/{stream_id}/frames/"
                        if db_stream:
                            db_stream.frames_s3_path = frames_s3_path
                            db_session.commit()
                    
                    # Analyze the frame
                    detection = detect_objects(frame_path)
                    
                    # Add to vector store
                    session.frame_knowledge.append(Document(page_content=detection, metadata={"frame": frame_id, "timestamp": timestamp}))
                    
                    # Update vector store periodically
                    if len(session.frame_knowledge) % 5 == 0 or len(session.frame_knowledge) == 1:
                        session.current_vectorstore = FAISS.from_documents(session.frame_knowledge, embedding)
                        
                        # Save vectorstore to S3 periodically
                        if db_session and db_stream:
                            faiss_dir = os.path.join(session.temp_dir, f"{stream_id}_faiss")
                            os.makedirs(faiss_dir, exist_ok=True)
                            session.current_vectorstore.save_local(faiss_dir)
                            index_s3_key = upload_to_s3(os.path.join(faiss_dir, "index.faiss"), "vectorstore/index.faiss", user_id, stream_id)
                            pkl_s3_key = upload_to_s3(os.path.join(faiss_dir, "index.pkl"), "vectorstore/index.pkl", user_id, stream_id)
                            
                            if index_s3_key and pkl_s3_key and not db_stream.vectorstore_s3_path:
                                db_stream.vectorstore_s3_path = f"users/{user_id}/{stream_id}/vectorstore/"
                                db_session.commit()
                        
                        with open(log_file, 'a') as f:
                            f.write(f"Updated vector store with {len(session.frame_knowledge)} frames\n")
                    
                    # Track entities for pattern detection (similar to video processing)
                    for line in detection.lower().split("\n"):
                        if not line.strip():
                            continue
                        
                        # People
                        if "person" in line or "man" in line or "woman" in line or "individual" in line:
                            descriptors = []
                            colors = ["red", "blue", "green", "white", "black", "yellow", "orange", "gray", "dark", "light"]
                            for color in colors:
                                if color in line:
                                    descriptors.append(color)
                            
                            clothing_items = ["shirt", "jacket", "coat", "hat", "cap", "hoodie", "pants", "jeans", "shorts"]
                            for item in clothing_items:
                                if item in line:
                                    descriptors.append(item)
                            
                            entity_type = "person"
                            if "man" in line: entity_type = "man"
                            if "woman" in line: entity_type = "woman"
                            
                            entity_key = f"{' '.join(descriptors)} {entity_type}"
                            
                            if entity_key.strip() in tracked_entities:
                                tracked_entities[entity_key.strip()]["frames"].append(frame_id)
                                tracked_entities[entity_key.strip()]["descriptions"].append(line)
                            else:
                                tracked_entities[entity_key.strip()] = {
                                    "frames": [frame_id],
                                    "descriptions": [line]
                                }
                        
                        # Vehicles
                        if "car" in line or "vehicle" in line or "motorcycle" in line or "truck" in line:
                            colors = ["red", "blue", "green", "white", "black", "silver", "gray", "yellow", "orange"]
                            color_matches = [color for color in colors if color in line]
                            vehicle_type = 'car' if 'car' in line else 'motorcycle' if 'motorcycle' in line else 'truck' if 'truck' in line else 'vehicle'
                            entity_key = f"{' '.join(color_matches)} {vehicle_type}"
                            
                            if entity_key.strip() in tracked_entities:
                                tracked_entities[entity_key.strip()]["frames"].append(frame_id)
                                tracked_entities[entity_key.strip()]["descriptions"].append(line)
                            else:
                                tracked_entities[entity_key.strip()] = {
                                    "frames": [frame_id],
                                    "descriptions": [line]
                                }
                    
                    # Suspicious activity detection
                    is_suspicious, suspicious_reason = is_genuinely_suspicious(detection, frame_id, tracked_entities)
                    
                    # Log and alert if suspicious
                    if is_suspicious:
                        alert_count += 1
                        
                        # Log the alert with email notification
                        log_stream_alert(
                            frame_id, 
                            detection, 
                            suspicious_reason, 
                            user_id, 
                            session, 
                            db_session=db_session, 
                            db_stream=db_stream, 
                            user_email=user_email,
                            timestamp=timestamp
                        )
                        
                        with open(log_file, 'a') as f:
                            f.write(f"[ALERT] Frame {frame_id}: {suspicious_reason}\nDetails: {detection}\n")
                        
                        # Add alert to database
                        if db_session and db_stream:
                            new_alert = StreamAlert(
                                stream_id=db_stream.id,
                                frame_id=frame_id,
                                timestamp=timestamp,
                                description=f"{suspicious_reason}\n{detection}",
                                is_confirmed_alert=True
                            )
                            db_session.add(new_alert)
                            db_session.commit()
                            
                            # Update alert count in the stream record
                            db_stream.alert_count = alert_count
                            db_session.commit()
                            
                            # Add to confirmed alerts list
                            confirmed_alerts.append(new_alert.id)
                    else:
                        # Get a brief description from the detection text
                        brief_description = detection.split("\n")[0] if "\n" in detection else detection[:100]
                        
                        # Check if the content is similar to the previous frame
                        is_similar = False
                        if session.last_detection:
                            similarity_words = set(session.last_detection.lower().split()) & set(detection.lower().split())
                            similarity_ratio = len(similarity_words) / max(1, len(set(detection.lower().split())))
                            is_similar = similarity_ratio > 0.8
                        
                        if is_similar:
                            session.static_frame_count += 1
                        else:
                            session.static_frame_count = 0
                        
                        # Store current detection for next comparison
                        session.last_detection = detection
                        
                        # Log based on whether we have static frames
                        if session.static_frame_count >= 3:  # Same threshold as video processing
                            log_message = f"Frame {frame_id} (Time: {timestamp}s): No new movements - {brief_description}"
                        else:
                            log_message = f"Frame {frame_id} (Time: {timestamp}s): {brief_description}"
                        
                        logs.append(log_message)
                        with open(log_file, 'a') as f:
                            f.write(log_message + "\n")
                        
                        # Add to logs database
                        if db_session and db_stream:
                            new_log = StreamLog(
                                stream_id=db_stream.id,
                                frame_id=frame_id,
                                timestamp=timestamp,
                                message=brief_description
                            )
                            db_session.add(new_log)
                            
                            # Commit every few frames to avoid too many transactions
                            if frame_id % 10 == 0:
                                db_session.commit()
                
                except Exception as e:
                    with open(log_file, 'a') as f:
                        f.write(f"Error processing frame {frame_id}: {str(e)}\n")
                    continue
            
            frame_id += 1
            
        # Final database updates
        logs_s3_key = upload_to_s3(log_file, f"logs/stream_log.txt", user_id, stream_id)
        
        if db_session and db_stream:
            db_stream.status = "inactive"
            db_stream.logs_s3_path = logs_s3_key
            db_stream.processed_frames = frame_id
            db_session.commit()
        
        # At end of stream processing or periodically, save logs to S3
        try:
            # Save logs to a JSON file
            logs_json_file = os.path.join(session.temp_dir, f"{stream_id}_logs.json")
            
            # Format logs for storage
            json_logs = []
            for log_message in logs:
                # Parse the log message to extract frame_id and timestamp
                frame_id = None
                timestamp = None
                message = log_message
                
                # Try to extract frame_id and timestamp from log message
                if "Frame " in log_message and " (Time: " in log_message and "s):" in log_message:
                    try:
                        frame_part = log_message.split("Frame ")[1].split(" (Time:")[0]
                        time_part = log_message.split(" (Time: ")[1].split("s):")[0]
                        message_part = log_message.split("s): ")[1] if "s): " in log_message else log_message
                        
                        frame_id = int(frame_part.strip())
                        timestamp = float(time_part.strip())
                        message = message_part.strip()
                    except:
                        # If parsing fails, keep the original message
                        pass
                
                json_logs.append({
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "message": message,
                    "log_type": "info",
                    "created_at": datetime.now().isoformat()
                })
            
            # Write to JSON file
            with open(logs_json_file, 'w') as f:
                json.dump({"logs": json_logs}, f)
            
            # Upload to S3
            logs_s3_key = upload_to_s3(logs_json_file, "stream_logs/stream_logs.json", user_id, stream_id)
            
            # Update database with S3 path
            if logs_s3_key and db_session and db_stream:
                logs_s3_path = f"users/{user_id}/{stream_id}/stream_logs"
                db_stream.logs_s3_path = logs_s3_path
                db_session.commit()
                
                print(f"Logs saved to S3: {logs_s3_path}")
        except Exception as e:
            print(f"Error saving logs to S3: {str(e)}")
            with open(log_file, 'a') as f:
                f.write(f"Error saving logs to S3: {str(e)}\n")
        
    except Exception as e:
        print(f"Error in stream processing: {str(e)}")
        with open(log_file, 'a') as f:
            f.write(f"Stream processing error: {str(e)}\n")
        
        if db_session and db_stream:
            db_stream.status = "error"
            db_session.commit()
    finally:
        # Release the capture
        cap.release()
        session.processing = False
        
        # Final log
        with open(log_file, 'a') as f:
            f.write(f"Stream processing stopped. Processed {frame_id} frames.\n")
        
        # Upload final logs to S3
        logs_s3_key = upload_to_s3(log_file, f"logs/stream_log.txt", user_id, stream_id)
        
        # Final database update
        if db_session and db_stream:
            db_stream.status = "inactive"
            db_stream.logs_s3_path = logs_s3_key
            db_stream.processed_frames = frame_id
            db_session.commit()
    
    return True


@router.post("/register", response_model=StreamSchema)
async def register_stream(
    stream: StreamCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Register a new stream for monitoring"""
    # Check if a stream with this ID already exists
    existing_stream = db.query(Stream).filter(Stream.stream_id == stream.stream_id).first()
    if existing_stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stream with ID {stream.stream_id} already exists"
        )
    
    # Create new stream record
    db_stream = Stream(
        stream_id=stream.stream_id,
        name=stream.name,
        ivs_url=stream.ivs_url,
        status="inactive",
        user_id=current_user.id
    )
    db.add(db_stream)
    db.commit()
    db.refresh(db_stream)
    
    return db_stream


@router.post("/{stream_id}/start", response_model=dict)
async def start_stream(
    stream_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Start processing a registered stream"""
    # Find the stream in the database
    db_stream = db.query(Stream).filter(
        Stream.stream_id == stream_id,
        Stream.user_id == current_user.id
    ).first()
    
    if not db_stream:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream with ID {stream_id} not found"
        )
    
    # Check if the stream is already being processed
    if user_id_key := current_user.id:
        if user_id_key not in streaming_sessions:
            streaming_sessions[user_id_key] = {}
        
        if stream_id in streaming_sessions[user_id_key] and streaming_sessions[user_id_key][stream_id].processing:
            return {"message": "Stream is already being processed", "stream_id": stream_id}
        
        # Create a new streaming session
        streaming_sessions[user_id_key][stream_id] = StreamSession(
            user_id=current_user.id,
            stream_id=stream_id,
            ivs_url=db_stream.ivs_url
        )
        
        # Start processing in background
        background_tasks.add_task(process_stream, stream_id, current_user.id, db)
        
        # Update stream status
        db_stream.status = "active"
        db_stream.last_active_at = datetime.now()
        db.commit()
        
        return {"message": "Stream processing started", "stream_id": stream_id}
    
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to start stream processing"
    )


@router.post("/{stream_id}/stop", response_model=dict)
async def stop_stream(
    stream_id: str,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Stop processing a stream"""
    # Find the stream in the database
    db_stream = db.query(Stream).filter(
        Stream.stream_id == stream_id,
        Stream.user_id == current_user.id
    ).first()
    
    if not db_stream:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream with ID {stream_id} not found"
        )
    
    # Check if the stream is being processed
    if user_id_key := current_user.id:
        if user_id_key in streaming_sessions and stream_id in streaming_sessions[user_id_key]:
            streaming_sessions[user_id_key][stream_id].processing = False
            
            # Update stream status
            db_stream.status = "inactive"
            db.commit()
            
            return {"message": "Stream processing stopped", "stream_id": stream_id}
    
    return {"message": "Stream was not being processed", "stream_id": stream_id}


@router.get("/status/{stream_id}", response_model=dict)
async def get_stream_status(
    stream_id: str,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Get the current status of a stream"""
    # Find the stream in the database
    db_stream = db.query(Stream).filter(
        Stream.stream_id == stream_id,
        Stream.user_id == current_user.id
    ).first()
    
    if not db_stream:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream with ID {stream_id} not found"
        )
    
    # Check if the stream is in memory
    in_memory = False
    processing = False
    frame_count = 0
    
    if user_id_key := current_user.id:
        if user_id_key in streaming_sessions and stream_id in streaming_sessions[user_id_key]:
            session = streaming_sessions[user_id_key][stream_id]
            in_memory = True
            processing = session.processing
            frame_count = len(session.frame_knowledge)
    
    return {
        "stream_id": stream_id,
        "name": db_stream.name,
        "status": db_stream.status,
        "alert_count": db_stream.alert_count,
        "processed_frames": db_stream.processed_frames,
        "in_memory": in_memory,
        "processing": processing,
        "frame_count": frame_count,
        "last_active": db_stream.last_active_at
    }


@router.get("/all", response_model=List[StreamSchema])
async def get_all_streams(
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Get all streams for the current user"""
    streams = db.query(Stream).filter(Stream.user_id == current_user.id).all()
    return streams


@router.get("/{stream_id}", response_model=StreamDetail)
async def get_stream_details(
    stream_id: str,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific stream"""
    stream = db.query(Stream).filter(
        Stream.stream_id == stream_id,
        Stream.user_id == current_user.id
    ).first()
    
    if not stream:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream with ID {stream_id} not found"
        )
    
    return stream


@router.post("/{stream_id}/chat", response_model=dict)
async def chat_with_stream(
    stream_id: str,
    query: Dict[str, Any],
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Chat with the AI about a stream's content"""
    # Find the stream in the database
    db_stream = db.query(Stream).filter(
        Stream.stream_id == stream_id,
        Stream.user_id == current_user.id
    ).first()
    
    if not db_stream:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream with ID {stream_id} not found"
        )
    
    # Try to find in memory first
    if user_id_key := current_user.id:
        if user_id_key in streaming_sessions and stream_id in streaming_sessions[user_id_key]:
            session = streaming_sessions[user_id_key][stream_id]
            
            if len(session.frame_knowledge) > 0:
                # If vectorstore is available
                if session.current_vectorstore:
                    language_model = ChatOpenAI(model_name="gpt-4o", temperature=0)
                    
                    custom_prompt = PromptTemplate.from_template(
                        """You are analyzing a live video stream by reviewing image frames extracted from it. 
Each frame was analyzed to detect objects like humans, cars, their colors, and activities. Use all frame descriptions 
to answer questions about what's happening in the stream. You can summarize patterns, behavior, and suspicious activity.

Context:
{context}

Question: {question}
Answer:"""
                    )
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=language_model,
                        chain_type="stuff",
                        retriever=session.current_vectorstore.as_retriever(),
                        chain_type_kwargs={"prompt": custom_prompt},
                        return_source_documents=False
                    )
                    
                    response = qa_chain.invoke({"query": query.get("question", "")})
                    return {"answer": response["result"]}
                
                # If no vectorstore but we have frame knowledge
                frames_context = "\n".join([
                    f"Frame {doc.metadata['frame']} (Time: {doc.metadata.get('timestamp', 0)}s): {doc.page_content}" 
                    for doc in session.frame_knowledge
                ])
                
                language_model = ChatOpenAI(model_name="gpt-4o", temperature=0)
                fallback_response = language_model.invoke(
                    f"""You are analyzing a live video stream based on these frame descriptions:
                    
                    {frames_context[:4000]}
                    
                    Question: {query.get("question", "")}
                    
                    Please answer based on the available information.
                    """
                )
                
                return {"answer": fallback_response.content}
    
    # If not in memory, try to load from S3
    if db_stream.vectorstore_s3_path:
        try:
            # Similar to video.py implementation - load vectorstore from S3
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            faiss_dir = os.path.join(temp_dir, "faiss_index")
            os.makedirs(faiss_dir, exist_ok=True)
            
            # Try to download vectorstore files
            index_s3_path = f"users/{current_user.id}/{stream_id}/vectorstore/index.faiss"
            pkl_s3_path = f"users/{current_user.id}/{stream_id}/vectorstore/index.pkl"
            
            # Configure S3 client from env vars
            s3 = boto3.client('s3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
                region_name=os.getenv("AWS_REGION"))
            bucket_name = os.getenv("BUCKET_NAME")
            
            # Download files
            s3.download_file(bucket_name, index_s3_path, os.path.join(faiss_dir, "index.faiss"))
            s3.download_file(bucket_name, pkl_s3_path, os.path.join(faiss_dir, "index.pkl"))
            
            # Load vectorstore
            embedding = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(faiss_dir, embedding, allow_dangerous_deserialization=True)
            
            # Create QA chain
            language_model = ChatOpenAI(model_name="gpt-4o", temperature=0)
            custom_prompt = PromptTemplate.from_template(
                """You are analyzing a live video stream by reviewing image frames extracted from it. 
Each frame was analyzed to detect objects like humans, cars, their colors, and activities. Use all frame descriptions 
to answer questions about what's happening in the stream. You can summarize patterns, behavior, and suspicious activity.

Context:
{context}

Question: {question}
Answer:"""
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=language_model,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": custom_prompt},
                return_source_documents=False
            )
            
            # Get answer
            response = qa_chain.invoke({"query": query.get("question", "")})
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            return {"answer": response["result"]}
            
        except Exception as e:
            return {"answer": f"Error loading stream data: {str(e)}"}
    
    # No data available
    return {"answer": "No data available for this stream yet."}


@router.delete("/{stream_id}", response_model=dict)
async def delete_stream(
    stream_id: str,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Delete a stream and all its data"""
    # Find the stream in the database
    db_stream = db.query(Stream).filter(
        Stream.stream_id == stream_id,
        Stream.user_id == current_user.id
    ).first()
    
    if not db_stream:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream with ID {stream_id} not found"
        )
    
    # Stop stream processing if active
    if user_id_key := current_user.id:
        if user_id_key in streaming_sessions and stream_id in streaming_sessions[user_id_key]:
            # Stop processing
            streaming_sessions[user_id_key][stream_id].processing = False
            
            # Clean up temp files
            streaming_sessions[user_id_key][stream_id].cleanup()
            
            # Remove from sessions dictionary
            del streaming_sessions[user_id_key][stream_id]
    
    # Delete S3 objects if base path exists
    s3_base_path = f"users/{current_user.id}/{stream_id}/"
    try:
        # Configure S3 client from env vars
        s3 = boto3.client('s3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
            region_name=os.getenv("AWS_REGION"))
        bucket_name = os.getenv("BUCKET_NAME")
        
        # List all objects in the stream folder
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_base_path)
        
        # Delete all objects
        if 'Contents' in response:
            for obj in response['Contents']:
                s3.delete_object(Bucket=bucket_name, Key=obj['Key'])
    except Exception as e:
        print(f"Error deleting S3 objects: {str(e)}")
    
    # Delete from database
    db.delete(db_stream)
    db.commit()
    
    return {"message": f"Stream {stream_id} deleted successfully"}


@router.post("/{stream_id}/update", response_model=dict)
async def update_stream(
    stream_id: str,
    stream_data: dict,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """
    Update stream information
    """
    # Find the stream in DB
    db_stream = db.query(Stream).filter(
        Stream.stream_id == stream_id,
        Stream.user_id == current_user.id
    ).first()
    
    if not db_stream:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream with ID {stream_id} not found"
        )
    
    # Update fields if provided
    if "name" in stream_data and stream_data["name"]:
        db_stream.name = stream_data["name"]
        
    if "ivs_url" in stream_data and stream_data["ivs_url"]:
        db_stream.ivs_url = stream_data["ivs_url"]
    
    # Update timestamp
    db_stream.updated_at = datetime.now()
    
    # Save to database
    db.commit()
    db.refresh(db_stream)
    
    # Clear any existing session if the URL changed
    if "ivs_url" in stream_data and stream_id in streaming_sessions.get(current_user.id, {}):
        try:
            # Stop any active processing
            streaming_session = streaming_sessions[current_user.id][stream_id]
            streaming_session.cleanup()
            del streaming_sessions[current_user.id][stream_id]
            print(f"Cleared streaming session for {stream_id} due to URL update")
        except Exception as e:
            print(f"Error clearing streaming session: {e}")
    
    return {"message": f"Stream {stream_id} updated successfully"}


@router.get("/{stream_id}/logs", response_model=dict)
async def get_stream_logs(
    stream_id: str,
    current_user: User = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    """Get logs for a specific stream, either from database or S3"""
    # Find the stream in the database
    db_stream = db.query(Stream).filter(
        Stream.stream_id == stream_id,
        Stream.user_id == current_user.id
    ).first()
    
    if not db_stream:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream with ID {stream_id} not found"
        )
    
    # Retrieve logs - first try from database
    logs = db.query(StreamLog).filter(
        StreamLog.stream_id == db_stream.id
    ).order_by(StreamLog.timestamp.desc()).limit(200).all()
    
    # If we don't have logs in the database, check if we have them in S3
    if not logs and db_stream.logs_s3_path:
        try:
            # Configure S3 client from env vars
            s3 = boto3.client('s3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
                region_name=os.getenv("AWS_REGION"))
            bucket_name = os.getenv("BUCKET_NAME")
            
            # Construct the full S3 path
            logs_s3_key = f"{db_stream.logs_s3_path}/stream_logs.json"
            
            # Download logs file from S3
            try:
                temp_dir = tempfile.mkdtemp()
                temp_file = os.path.join(temp_dir, "stream_logs.json")
                
                s3.download_file(bucket_name, logs_s3_key, temp_file)
                
                # Read logs from file
                with open(temp_file, 'r') as f:
                    logs_data = json.load(f)
                
                # Convert to a format similar to what the database would return
                logs = []
                for log_entry in logs_data.get('logs', []):
                    logs.append({
                        'frame_id': log_entry.get('frame_id'),
                        'timestamp': log_entry.get('timestamp'),
                        'message': log_entry.get('message'),
                        'log_type': log_entry.get('log_type', 'info'),
                        'created_at': log_entry.get('created_at')
                    })
                
                # Clean up
                shutil.rmtree(temp_dir)
                
            except Exception as e:
                print(f"Error downloading logs from S3: {str(e)}")
                return {"logs": [], "error": f"Failed to load stream logs: {str(e)}"}
                
        except Exception as e:
            print(f"Error accessing S3 for logs: {str(e)}")
            return {"logs": [], "error": f"Failed to load stream logs: {str(e)}"}
    
    # Format the logs for the response
    formatted_logs = []
    for log in logs:
        formatted_logs.append({
            "id": log.id if hasattr(log, 'id') else None,
            "frame_id": log.frame_id,
            "timestamp": log.timestamp,
            "message": log.message,
            "log_type": log.log_type,
            "created_at": log.created_at.isoformat() if hasattr(log, 'created_at') and log.created_at else None
        })
    
    return {"logs": formatted_logs} 