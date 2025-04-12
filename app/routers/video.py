import os
import shutil
import tempfile
import json
import boto3
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from ..auth import get_current_active_user
from ..models.user import User
from ..models.video import Video, VideoLog, VideoAlert
from ..schemas.video import VideoCreate, VideoUpdate, Video as VideoSchema, VideoDetail, VideoAlert as VideoAlertSchema
from ..database import get_db
from ..utils.email import send_alert_email

# Load environment variables
load_dotenv()

# AWS S3 Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Initialize S3 client
s3 = boto3.client('s3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION)

# Video analysis code (imported from provided code)
import cv2
import base64
import threading
import time
from collections import defaultdict
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import openai

# Set OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = openai.api_key

# LangChain Config
embedding = OpenAIEmbeddings()
language_model = ChatOpenAI(model_name="gpt-4o", temperature=0)

custom_prompt = PromptTemplate.from_template(
    """You are a video analysis assistant. You are analyzing a video by reviewing many image frames extracted from it. 
Each frame was analyzed to detect objects like humans, cars, their colors, and brands. Use all frame descriptions 
to answer questions about the entire video. You can summarize patterns, repeated behavior, and suspicious activity.

Context:
{context}

Question: {question}
Answer:"""
)

# Global variables for each user session
user_sessions = {}

router = APIRouter(
    prefix="/video",
    tags=["video"],
)

class UserSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.videos = {}  # Dictionary to track multiple videos: video_id -> VideoData
        self.current_video_id = None
        self.temp_dir = tempfile.mkdtemp()
        
    def cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

class VideoData:
    def __init__(self, video_id):
        self.video_id = video_id
        self.frame_knowledge = []
        self.current_vectorstore = None
        self.alerted_descriptions = set()
        self.object_memory = defaultdict(list)
        self.processing = False
        self.video_path = None
        self.filename = None
        self.progress = 0  # Processing progress (0-100%)
        self.total_frames = 0  # Total frames in the video
        self.processed_frames = 0  # Number of processed frames

def get_s3_url(s3_key):
    """Generate a URL for an S3 object"""
    try:
        return f"https://{BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
    except Exception as e:
        print(f"Error generating S3 URL: {e}")
        return None

def upload_to_s3(local_path, s3_key, user_id, video_id):
    try:
        full_key = f"users/{user_id}/{video_id}/{s3_key}"
        s3.upload_file(local_path, BUCKET_NAME, full_key)
        return full_key
    except Exception as e:
        print(f"Failed to upload {s3_key} to S3: {e}")
        return None

def log_alert(frame_id, description, suspicious_reason, user_id, video_data, session, db_session=None, db_video=None, user_email=None, fps=None):
    """
    Log a suspicious activity alert
    
    Args:
        frame_id: The frame ID where the suspicious activity was detected
        description: The description of what was detected in the frame
        suspicious_reason: The specific reason why this activity is considered suspicious
        user_id: User ID
        video_data: VideoData object
        session: User session
        db_session: Database session for retrieving user info
        db_video: Video database object
        user_email: User's email address
        fps: Frames per second for timestamp calculation
    """
    alert_key = f"{frame_id}_{suspicious_reason[:20]}"  # Create a key to avoid duplicate alerts
    
    if alert_key not in video_data.alerted_descriptions:
        alert_file = os.path.join(session.temp_dir, f"{video_data.video_id}_alerts.txt")
        
        # Extract confidence information from the suspicious reason
        confidence_level = "HIGH" if any(keyword in suspicious_reason.lower() for keyword in [
            "weapon", "gun", "knife", "assault", "breaking", "forced entry", "theft"
        ]) else "MEDIUM"
        
        # Log the alert
        alert_message = f"[ALERT-{confidence_level}] Frame {frame_id}: {suspicious_reason}\nDetails: {description}"
        with open(alert_file, "a") as f:
            f.write(alert_message + "\n\n")
            
        alert_s3_key = upload_to_s3(alert_file, f"logs/alerts.txt", user_id, video_data.video_id)
        video_data.alerted_descriptions.add(alert_key)
        
        # Regardless of confidence level, always try to send an email alert if we have user email
        # This ensures we don't miss critical security events
        try:
            if user_email and db_video:
                # Calculate timestamp in seconds
                timestamp = int(frame_id / fps) if fps and fps > 0 else 0
                
                # Try to get thumbnail URL for the frame
                frame_thumbnail_url = None
                try:
                    frame_s3_path = f"users/{user_id}/{video_data.video_id}/frames/frame_{frame_id}.jpg"
                    frame_thumbnail_url = get_s3_url(frame_s3_path)
                except Exception as e:
                    print(f"Could not get thumbnail URL: {str(e)}")
                    
                # Log that we're attempting to send an email alert
                print(f"Sending alert email to {user_email} for suspicious activity in video {db_video.filename}")
                
                # Send email alert
                email_sent = send_alert_email(
                    user_email, 
                    db_video.filename, 
                    description, 
                    frame_id, 
                    timestamp, 
                    frame_thumbnail_url
                )
                
                # Log whether the email was sent successfully
                if email_sent:
                    print(f"Alert email sent successfully to {user_email}")
                    with open(alert_file, "a") as f:
                        f.write(f"[EMAIL ALERT SENT] Email notification sent to {user_email}\n\n")
                else:
                    print(f"Failed to send alert email to {user_email}")
                    with open(alert_file, "a") as f:
                        f.write(f"[EMAIL ALERT FAILED] Could not send email notification to {user_email}\n\n")
        except Exception as e:
            print(f"Exception when sending alert email: {str(e)}")
            with open(alert_file, "a") as f:
                f.write(f"[EMAIL ALERT ERROR] {str(e)}\n\n")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def detect_objects(image_path):
    base64_image = encode_image(image_path)
    prompt = [
        {"type": "text", "text": "You are a security surveillance analyst. Analyze this image and provide a detailed description. Focus on people, vehicles, and their actions. ONLY mark something as 'SUSPICIOUS:YES' at the end of your analysis if it shows clear evidence of: 1) unauthorized access, 2) theft/tampering, 3) hiding/concealment behavior, 4) weapon possession, 5) physical threats, 6) breaking into property, or 7) vandalism. Being specific is crucial. Otherwise, mark it as 'SUSPICIOUS:NO'. Regular activities like walking, standing, using phones, talking, or normal vehicle movement are NOT suspicious. Do NOT overreact to ambiguous scenarios."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a high-precision security analyst with a very low false positive rate. Your analysis should be detailed and factual. Never mark something as suspicious unless there is clear visual evidence of a genuine security threat. Avoid speculation."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message['content']

def is_genuinely_suspicious(detection_text, frame_id, entity_patterns):
    """
    Sophisticated suspicious activity detector with multiple layers of verification
    
    Args:
        detection_text: The text description from GPT-4o
        frame_id: Current frame ID
        entity_patterns: Dictionary tracking entity appearances over time
        
    Returns:
        (is_suspicious, reason): Tuple with boolean indicating if activity is suspicious and reason why
    """
    # Quick rejection: if GPT-4o explicitly marked as NOT suspicious, return False immediately
    if "SUSPICIOUS:NO" in detection_text.upper() or "SUSPICIOUS: NO" in detection_text.upper():
        return False, "Not suspicious"
    
    # Only continue analysis if GPT-4o flagged as suspicious
    gpt_flagged = "SUSPICIOUS:YES" in detection_text.upper() or "SUSPICIOUS: YES" in detection_text.upper()
    
    if not gpt_flagged:
        return False, "Not flagged as suspicious by analysis"
    
    # Define highly specific suspicious behaviors - only consider these when GPT has flagged suspicious
    high_confidence_keywords = [
        "breaking window", "smashing", "stealing", "weapon", "gun", "knife",
        "forced entry", "breaking in", "jimmying", "lockpicking", 
        "masked figure", "concealed face", "threatening posture", "assault",
        "vandalism", "physical altercation", "trespassing", "tailgating",
        "hostage", "unauthorized access"
    ]
    
    # Specific suspicious actions
    explicit_suspicious_behavior = any(keyword in detection_text.lower() for keyword in high_confidence_keywords)
    
    # Calculate suspicious patterns based on entity tracking
    suspicious_pattern = False
    pattern_reason = ""
    
    # Extract suspicious patterns from entity tracking
    if entity_patterns:
        for entity, data in entity_patterns.items():
            # Only consider entities that have appeared multiple times
            if len(data["frames"]) >= 3:
                # Check for gap patterns (vehicle or person coming and going)
                time_gaps = [data["frames"][i+1] - data["frames"][i] for i in range(len(data["frames"])-1)]
                
                # Check if the entity is exhibiting stalking/surveillant behavior (coming and going)
                if any(gap > 100 for gap in time_gaps) and "person" in entity:
                    suspicious_pattern = True
                    pattern_reason = f"Person {entity} showing stalking behavior (repeated appearances with long gaps)"
                
                # Check for vehicles repeatedly appearing/disappearing
                elif any(gap > 150 for gap in time_gaps) and ("car" in entity or "vehicle" in entity) and len(data["frames"]) >= 4:
                    suspicious_pattern = True
                    pattern_reason = f"Vehicle {entity} making multiple suspicious visits"
    
    # Determine the final verdict - require both GPT flagging AND supporting evidence
    is_suspicious = gpt_flagged and (explicit_suspicious_behavior or suspicious_pattern)
    
    if is_suspicious:
        if explicit_suspicious_behavior:
            reason = "Clear suspicious activity detected matching known threat patterns"
        elif suspicious_pattern:
            reason = f"Suspicious pattern detected: {pattern_reason}"
    else:
        reason = "Insufficient evidence of threat (requires both AI flagging and supporting evidence)"
    
    return is_suspicious, reason

def process_video(video_path, user_id, video_id, db_session=None):
    if user_id not in user_sessions:
        print(f"User session {user_id} not found")
        return False
        
    session = user_sessions[user_id]
    if video_id not in session.videos:
        print(f"Video {video_id} not found for user {user_id}")
        return False
        
    video_data = session.videos[video_id]
    video_data.processing = True
    
    # Update video status in DB
    db_video = None
    if db_session:
        db_video = db_session.query(Video).filter(Video.video_id == video_id).first()
        if db_video:
            db_video.status = "processing"
            db_session.commit()
    
    # Create directories
    frames_dir = os.path.join(session.temp_dir, f"{video_id}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    log_file = os.path.join(session.temp_dir, f"{video_id}_log.txt")
    
    # Check if we're resuming - if log file exists
    resuming = os.path.exists(log_file)
    last_processed_frame = 0
    
    if resuming:
        # Check for processed frames in database
        if db_video and db_video.processed_frames:
            last_processed_frame = db_video.processed_frames
            with open(log_file, 'a') as f:
                f.write(f"Resuming processing from frame {last_processed_frame}...\n")
    else:
        # Start fresh log
        with open(log_file, 'w') as f:
            f.write(f"Starting video processing for {video_id}...\n")
    
    logs = []
    
    # Activity tracking for context awareness
    tracked_entities = {}  # Format: {entity_id: {"frames": [frame_ids], "descriptions": [descriptions]}}
    
    # Add tracking for consecutive similar frames
    last_detection = ""
    static_frame_count = 0
    max_static_frames = 3  # Number of consecutive similar frames before considering "no new movements"
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    FRAME_INTERVAL = 30  # Analyze every 30th frame
    
    # Get video metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Set total frames to track progress
    video_data.total_frames = frame_count
    
    # Update DB with total frames
    if db_session and db_video:
        db_video.total_frames = frame_count
        db_session.commit()
    
    alert_count = 0
    frames_s3_path = None
    thumbnail_path = None
    
    # Alert verification tracking
    potential_alerts = []  # Store potential alerts for verification
    confirmed_alerts = []  # Store only confirmed alerts
    
    # If resuming, skip to last processed frame
    if last_processed_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_processed_frame)
        frame_id = last_processed_frame
        video_data.processed_frames = last_processed_frame
        
        # Also look for any existing processed frames and knowledge in the session
        if db_video and db_video.vectorstore_s3_path:
            try:
                # Create temp directory for downloading vectorstore
                temp_dir = tempfile.mkdtemp()
                faiss_dir = os.path.join(temp_dir, "faiss_index")
                os.makedirs(faiss_dir, exist_ok=True)
                
                # Try to download vectorstore files from S3
                index_s3_path = f"users/{user_id}/{video_id}/vectorstore/index.faiss"
                pkl_s3_path = f"users/{user_id}/{video_id}/vectorstore/index.pkl"
                
                try:
                    # Download FAISS index file
                    index_local_path = os.path.join(faiss_dir, "index.faiss")
                    s3.download_file(BUCKET_NAME, index_s3_path, index_local_path)
                    
                    # Download metadata file
                    pkl_local_path = os.path.join(faiss_dir, "index.pkl")
                    s3.download_file(BUCKET_NAME, pkl_s3_path, pkl_local_path)
                    
                    # Load vectorstore
                    video_data.current_vectorstore = FAISS.load_local(faiss_dir, embedding, allow_dangerous_deserialization=True)
                    
                    # Clean up temp files
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Could not load existing vectorstore: {e}")
            except Exception as e:
                print(f"Error preparing for vectorstore download: {e}")
    
    # Periodically update database with progress (every 30 seconds)
    last_db_update_time = time.time()
    db_update_interval = 30  # seconds
    
    # Get user email for alerts
    user_email = None
    if db_session:
        # Get user associated with this video
        user = db_session.query(User).filter(User.id == user_id).first()
        if user:
            user_email = user.email
            print(f"Retrieved user email for alerts: {user_email}")
        else:
            print(f"Could not find user with ID {user_id} for email alerts")
    
    while cap.isOpened() and video_data.processing:
        try:
            ret, frame = cap.read()
            if not ret:
                break

            # Update progress (based on frame position)
            video_data.processed_frames = frame_id
            progress = min(99, int((frame_id / frame_count) * 100)) if frame_count > 0 else 0
            video_data.progress = progress
            
            # Update database periodically to allow resuming
            current_time = time.time()
            if db_session and db_video and (current_time - last_db_update_time) > db_update_interval:
                db_video.processed_frames = frame_id
                db_video.processing_progress = progress
                db_session.commit()
                last_db_update_time = current_time
            
            if frame_id % FRAME_INTERVAL == 0:
                frame_path = os.path.join(frames_dir, f"frame_{frame_id}.jpg")
                cv2.imwrite(frame_path, frame)
                
                # Save first frame as thumbnail
                if frame_id == 0:
                    thumbnail_path = frame_path
                
                # Log frame processing
                with open(log_file, 'a') as f:
                    f.write(f"Processing frame {frame_id} ({video_data.progress}%)...\n")
                
                try:
                    frame_s3_key = upload_to_s3(frame_path, f"frames/frame_{frame_id}.jpg", user_id, video_id)
                    if frame_id == 0 and frame_s3_key:
                        frames_s3_path = f"users/{user_id}/{video_id}/frames/"
                    
                    detection = detect_objects(frame_path)
                    
                    # Add to vector store
                    video_data.frame_knowledge.append(Document(page_content=detection, metadata={"frame": frame_id}))
                    
                    # Update vector store 
                    if len(video_data.frame_knowledge) % 5 == 0 or len(video_data.frame_knowledge) == 1:
                        video_data.current_vectorstore = FAISS.from_documents(video_data.frame_knowledge, embedding)
                        
                        # Save vectorstore to S3 periodically for resumability
                        if db_session and db_video:
                            faiss_dir = os.path.join(session.temp_dir, f"{video_id}_faiss")
                            os.makedirs(faiss_dir, exist_ok=True)
                            video_data.current_vectorstore.save_local(faiss_dir)
                            index_s3_key = upload_to_s3(os.path.join(faiss_dir, "index.faiss"), "vectorstore/index.faiss", user_id, video_id)
                            pkl_s3_key = upload_to_s3(os.path.join(faiss_dir, "index.pkl"), "vectorstore/index.pkl", user_id, video_id)
                            
                            if index_s3_key and pkl_s3_key and not db_video.vectorstore_s3_path:
                                db_video.vectorstore_s3_path = f"users/{user_id}/{video_id}/vectorstore/"
                                db_session.commit()
                        
                        with open(log_file, 'a') as f:
                            f.write(f"Updated vector store with {len(video_data.frame_knowledge)} frames\n")

                    # Track entities for better pattern detection
                    for line in detection.lower().split("\n"):
                        if not line.strip():
                            continue
                        
                        # Extract entities from the detection results
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
                            
                            # Create entity key
                            entity_type = "person"
                            if "man" in line: entity_type = "man"
                            if "woman" in line: entity_type = "woman"
                            
                            entity_key = f"{' '.join(descriptors)} {entity_type}"
                            
                            # Add to tracked entities
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
                    
                    # Improved suspicious activity detection with verification
                    is_suspicious, suspicious_reason = is_genuinely_suspicious(detection, frame_id, tracked_entities)
                    
                    # Log and alert only if truly suspicious
                    if is_suspicious:
                        alert_count += 1
                        alert_message = f"[ALERT] Frame {frame_id}: {suspicious_reason}\nDetails: {detection}"
                        
                        # Store alert information
                        potential_alerts.append({
                            "frame_id": frame_id,
                            "detection": detection,
                            "reason": suspicious_reason,
                            "timestamp": int(frame_id / fps) if fps > 0 else 0,
                            "frame_path": frame_path
                        })
                        
                        # Log the alert with email notification capability
                        log_alert(
                            frame_id, 
                            detection, 
                            suspicious_reason, 
                            user_id, 
                            video_data, 
                            session, 
                            db_session=db_session, 
                            db_video=db_video, 
                            user_email=user_email,
                            fps=fps
                        )
                        
                        with open(log_file, 'a') as f:
                            f.write(alert_message + "\n")
                        
                        # Only create an alert in DB if we're reasonably certain it's genuine
                        if db_session and db_video:
                            new_alert = VideoAlert(
                                video_id=db_video.id,
                                frame_id=frame_id,
                                timestamp=int(frame_id / fps) if fps > 0 else 0,
                                description=f"{suspicious_reason}\n{detection}",
                                is_confirmed_alert=True  # Mark as confirmed genuine alert
                            )
                            db_session.add(new_alert)
                            db_session.commit()
                                
                            # Add to confirmed alerts list
                            confirmed_alerts.append(new_alert.id)
                    else:
                        # Get a brief description from the detection text
                        brief_description = detection.split("\n")[0] if "\n" in detection else detection[:100]
                        
                        # Check if the content is similar to the previous frame
                        is_similar = False
                        if last_detection:
                            # Simple similarity check - can be improved with more sophisticated text comparison
                            similarity_words = set(last_detection.lower().split()) & set(detection.lower().split())
                            similarity_ratio = len(similarity_words) / max(1, len(set(detection.lower().split())))
                            is_similar = similarity_ratio > 0.8  # If 80% of words are the same, consider it similar
                        
                        if is_similar:
                            static_frame_count += 1
                        else:
                            static_frame_count = 0
                        
                        # Store current detection for next comparison
                        last_detection = detection
                        
                        # Log based on whether we have static frames
                        if static_frame_count >= max_static_frames:
                            log_message = f"Frame {frame_id}: No new movements - {brief_description}"
                        else:
                            log_message = f"Frame {frame_id}: {brief_description}"
                        
                        logs.append(log_message)
                        with open(log_file, 'a') as f:
                            f.write(log_message + "\n")

                except Exception as e:
                    with open(log_file, 'a') as f:
                        f.write(f"Error processing frame {frame_id}: {str(e)}\n")
                    continue

            frame_id += 1
        except Exception as e:
            with open(log_file, 'a') as f:
                f.write(f"Processing error: {str(e)}\n")
                
            # Update status to interrupted to allow resuming
            if db_session and db_video:
                db_video.status = "interrupted"
                db_video.processed_frames = frame_id
                db_video.processing_progress = video_data.progress
                db_session.commit()
                
            continue

    cap.release()
    
    # Set progress to 100% when finished
    video_data.progress = 100
    
    # Verification step: update alert count to reflect only confirmed alerts
    if db_session and db_video:
        accurate_alert_count = len(confirmed_alerts)
        
        # Update the alert count in the database
        db_video.alert_count = accurate_alert_count
    
    # Save logs
    with open(log_file, 'a') as f:
        f.write("Video processing complete!\n")
        f.write(f"Found {alert_count} potential security alerts, confirmed {len(confirmed_alerts)} genuine alerts\n")
        for log in logs:
            f.write(log + "\n")
    
    # Final upload of logs
    logs_s3_key = upload_to_s3(log_file, f"logs/video_log.txt", user_id, video_id)

    # Save and upload FAISS index if it exists
    vectorstore_s3_path = None
    if video_data.current_vectorstore:
        faiss_dir = os.path.join(session.temp_dir, f"{video_id}_faiss")
        os.makedirs(faiss_dir, exist_ok=True)
        video_data.current_vectorstore.save_local(faiss_dir)
        index_s3_key = upload_to_s3(os.path.join(faiss_dir, "index.faiss"), "vectorstore/index.faiss", user_id, video_id)
        pkl_s3_key = upload_to_s3(os.path.join(faiss_dir, "index.pkl"), "vectorstore/index.pkl", user_id, video_id)
        
        if index_s3_key and pkl_s3_key:
            vectorstore_s3_path = f"users/{user_id}/{video_id}/vectorstore/"
    
    # Upload thumbnail if available
    thumbnail_s3_key = None
    if thumbnail_path:
        thumbnail_s3_key = upload_to_s3(thumbnail_path, "thumbnail.jpg", user_id, video_id)
    
    # Update DB with completed status and paths
    if db_session and db_video:
        db_video.status = "completed"
        db_video.alert_count = len(confirmed_alerts)  # Use verified alert count
        db_video.frames_s3_path = frames_s3_path
        db_video.logs_s3_path = logs_s3_key
        db_video.vectorstore_s3_path = vectorstore_s3_path
        db_video.size_bytes = os.path.getsize(video_path) if os.path.exists(video_path) else None
        db_video.duration_seconds = int(duration)
        db_video.resolution = f"{width}x{height}"
        db_video.processed_frames = frame_id
        db_video.processing_progress = 100
        
        if thumbnail_s3_key:
            db_video.thumbnail_url = get_s3_url(thumbnail_s3_key)
        
        db_session.commit()
    
    video_data.processing = False
    return True

@router.post("/upload_and_analyze")
async def upload_and_analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    video_id: str = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Combined endpoint that handles both upload and analysis in one step.
    Also supports resuming analysis if the video was already uploaded.
    """
    # Check file size - limit to 200MB (200 * 1024 * 1024 bytes)
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB in bytes
    
    # Read a small part of the file to check content type
    file_head = await file.read(1024)
    # Seek back to the beginning of the file
    await file.seek(0)
    
    # Get file size
    file.file.seek(0, 2)  # Move to the end to get size
    file_size = file.file.tell()  # Get current position (file size)
    await file.seek(0)  # Reset to beginning
    
    # Check if file exceeds size limit
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the limit of 200MB. Your file is {file_size/(1024*1024):.2f}MB."
        )
    
    # Check if the video already exists in database
    existing_video = db.query(Video).filter(
        Video.video_id == video_id,
        Video.user_id == current_user.id
    ).first()
    
    # If video exists, check if we should resume analysis
    if existing_video:
        # Check if video file exists on S3
        if existing_video.video_s3_path and existing_video.status in ["uploaded", "interrupted"]:
            # Create user session if it doesn't exist
            if current_user.id not in user_sessions:
                user_sessions[current_user.id] = UserSession(current_user.id)
            
            session = user_sessions[current_user.id]
            
            # Create video data if it doesn't exist in memory
            if video_id not in session.videos:
                video_data = VideoData(video_id)
                video_data.filename = existing_video.filename
                
                # Download the video from S3 to local path
                try:
                    # Create temp directory for video
                    video_path = os.path.join(session.temp_dir, f"{video_id}_{existing_video.filename}")
                    video_data.video_path = video_path
                    
                    # Download video from S3
                    s3.download_file(BUCKET_NAME, existing_video.video_s3_path, video_path)
                    
                    # Add video to session
                    session.videos[video_id] = video_data
                    session.current_video_id = video_id
                    
                    # Update database status
                    existing_video.status = "processing"
                    db.commit()
                    
                    # Start processing video in background
                    background_tasks.add_task(process_video, video_path, current_user.id, video_id, db)
                    
                    return {
                        "message": "Resuming video analysis", 
                        "filename": existing_video.filename, 
                        "video_id": video_id,
                        "resumed": True
                    }
                    
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                        detail=f"Failed to resume analysis: {str(e)}"
                    )
            else:
                # Video already exists in session
                video_data = session.videos[video_id]
                
                # If it's already processing, just return status
                if video_data.processing:
                    return {
                        "message": "Video is already being processed",
                        "filename": existing_video.filename,
                        "video_id": video_id,
                        "resumed": False
                    }
                else:
                    # Start processing again
                    background_tasks.add_task(process_video, video_data.video_path, current_user.id, video_id, db)
                    return {
                        "message": "Resuming video analysis", 
                        "filename": existing_video.filename, 
                        "video_id": video_id,
                        "resumed": True
                    }
        elif existing_video.status == "completed":
            return {
                "message": "Video already processed", 
                "filename": existing_video.filename, 
                "video_id": video_id,
                "resumed": False
            }
    
    # If video doesn't exist or can't be resumed, process as a new upload
    
    # Create user session if it doesn't exist
    if current_user.id not in user_sessions:
        user_sessions[current_user.id] = UserSession(current_user.id)
    
    session = user_sessions[current_user.id]
    
    # Create new video data
    video_data = VideoData(video_id)
    video_data.filename = file.filename
    
    # Save uploaded video
    video_path = os.path.join(session.temp_dir, f"{video_id}_{file.filename}")
    video_data.video_path = video_path
    
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Add video to session
    session.videos[video_id] = video_data
    session.current_video_id = video_id
    
    # Upload original video to S3 directly in the video_id directory
    video_s3_key = upload_to_s3(video_path, file.filename, current_user.id, video_id)
    
    # Create entry in database
    s3_base_path = f"users/{current_user.id}/{video_id}/"
    
    # Create new video record or update existing one
    if existing_video:
        existing_video.filename = file.filename
        existing_video.s3_base_path = s3_base_path
        existing_video.status = "processing"
        existing_video.video_s3_path = video_s3_key
        db.commit()
        db_video = existing_video
    else:
        db_video = Video(
            video_id=video_id,
            filename=file.filename,
            s3_base_path=s3_base_path,
            status="processing",  # Set to processing immediately
            video_s3_path=video_s3_key,
            user_id=current_user.id,
            processed_frames=0,
            total_frames=0,
            processing_progress=0
        )
        db.add(db_video)
        db.commit()
        db.refresh(db_video)
    
    # Start processing video in background
    background_tasks.add_task(process_video, video_path, current_user.id, video_id, db)
    
    return {
        "message": "Video uploaded and analysis started", 
        "filename": file.filename, 
        "video_id": video_id, 
        "db_id": db_video.id,
        "resumed": False
    }

@router.post("/analyze/{filename}")
async def analyze_video(
    filename: str,
    background_tasks: BackgroundTasks,
    video_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    if current_user.id not in user_sessions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No video uploaded")
    
    session = user_sessions[current_user.id]
    
    # If no video_id specified, use current_video_id
    if not video_id:
        video_id = session.current_video_id
    
    if not video_id or video_id not in session.videos:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Video ID not found")
    
    video_data = session.videos[video_id]
    video_path = video_data.video_path
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video file not found")
    
    if video_data.processing:
        return {"message": "Video is already being processed"}
    
    # Start processing video in background
    background_tasks.add_task(process_video, video_path, current_user.id, video_id, db)
    
    return {"message": "Video analysis started", "video_id": video_id}

@router.get("/status")
async def get_status(current_user: User = Depends(get_current_active_user)):
    if current_user.id not in user_sessions:
        return {"processing": False, "has_data": False}
    
    session = user_sessions[current_user.id]
    
    # If no current video, return all false
    if not session.current_video_id or session.current_video_id not in session.videos:
        return {"processing": False, "has_data": False}
    
    video_data = session.videos[session.current_video_id]
    has_data = len(video_data.frame_knowledge) > 0
    
    return {
        "processing": video_data.processing, 
        "has_data": has_data, 
        "video_id": session.current_video_id,
        "progress": video_data.progress,
        "total_frames": video_data.total_frames,
        "processed_frames": video_data.processed_frames
    }

@router.post("/chat")
async def chat_with_analysis(
    query: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Get the video_id from the request
    video_id = query.get("video_id")
    
    if not video_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing video_id parameter")
    
    # Check if video exists in database
    db_video = db.query(Video).filter(
        Video.video_id == video_id,
        Video.user_id == current_user.id
    ).first()
    
    if not db_video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    
    # Try to get from memory cache first
    if current_user.id in user_sessions:
        session = user_sessions[current_user.id]
        
        # Check if video data is in memory
        if video_id in session.videos:
            video_data = session.videos[video_id]
            
            if len(video_data.frame_knowledge) > 0:
                # If vectorstore is available in memory, use it
                if video_data.current_vectorstore:
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=language_model,
                        chain_type="stuff",
                        retriever=video_data.current_vectorstore.as_retriever(),
                        chain_type_kwargs={"prompt": custom_prompt},
                        return_source_documents=False
                    )
                    
                    response = qa_chain.invoke({"query": query.get("question", "")})
                    return {"answer": response["result"]}
                
                # If no vectorstore but we have frame knowledge, use direct call
                frames_context = "\n".join([f"Frame {doc.metadata['frame']}: {doc.page_content}" 
                                        for doc in video_data.frame_knowledge])
                
                fallback_response = language_model.invoke(
                    f"""You are analyzing a video based on these frame descriptions:
                    
                    {frames_context[:4000]}
                    
                    Question: {query.get("question", "")}
                    
                    Please answer based on the available information.
                    """
                )
                
                return {"answer": fallback_response.content}
    
    # If video is not in memory or has no frame knowledge, try to load vectorstore from S3
    if db_video.vectorstore_s3_path:
        try:
            # Create temp directory for downloading vectorstore
            temp_dir = tempfile.mkdtemp()
            faiss_dir = os.path.join(temp_dir, "faiss_index")
            os.makedirs(faiss_dir, exist_ok=True)
            
            # Try to download vectorstore files from S3
            index_s3_path = f"users/{current_user.id}/{video_id}/vectorstore/index.faiss"
            pkl_s3_path = f"users/{current_user.id}/{video_id}/vectorstore/index.pkl"
            
            try:
                # Download FAISS index file
                index_local_path = os.path.join(faiss_dir, "index.faiss")
                s3.download_file(BUCKET_NAME, index_s3_path, index_local_path)
                
                # Download metadata file
                pkl_local_path = os.path.join(faiss_dir, "index.pkl")
                s3.download_file(BUCKET_NAME, pkl_s3_path, pkl_local_path)
                
                # Load vectorstore
                vectorstore = FAISS.load_local(faiss_dir, embedding, allow_dangerous_deserialization=True)
                
                # Create QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=language_model,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(),
                    chain_type_kwargs={"prompt": custom_prompt},
                    return_source_documents=False
                )
                
                # Get answer
                response = qa_chain.invoke({"query": query.get("question", "")})
                
                # Clean up temp files
                shutil.rmtree(temp_dir)
                
                return {"answer": response["result"]}
                
            except Exception as e:
                # Clean up temp files
                shutil.rmtree(temp_dir)
                print(f"Error loading vectorstore from S3: {e}")
                return {"answer": f"Unable to load video analysis data. Error: {str(e)}"}
        except Exception as e:
            print(f"Error creating temporary directory: {e}")
            return {"answer": "Unable to process your question at this time."}
    
    # If we get here, we couldn't get an answer
    return {"answer": "No analysis data is available for this video yet. Please wait for processing to complete or try analyzing the video again."}

@router.delete("/cleanup")
async def cleanup_session(current_user: User = Depends(get_current_active_user)):
    if current_user.id in user_sessions:
        session = user_sessions[current_user.id]
        
        # Stop all video processing
        for video_id, video_data in session.videos.items():
            video_data.processing = False
            
        session.cleanup()
        del user_sessions[current_user.id]
    
    return {"message": "Session cleaned up successfully"}

@router.get("/logs")
async def get_logs(current_user: User = Depends(get_current_active_user)):
    if current_user.id not in user_sessions:
        return {"logs": []}
    
    session = user_sessions[current_user.id]
    
    # If no current video, return empty logs
    if not session.current_video_id or session.current_video_id not in session.videos:
        return {"logs": []}
    
    video_data = session.videos[session.current_video_id]
    
    # Fetch logs from session
    logs = []
    
    # Add frame processing status
    if video_data.frame_knowledge:
        logs.append(f"Processed {len(video_data.frame_knowledge)} frames so far")
    
    # Add alerts if any exist
    if video_data.alerted_descriptions:
        logs.append("--- ALERTS ---")
        for idx, alert in enumerate(video_data.alerted_descriptions):
            logs.append(f"Alert {idx+1}: {alert}")
    
    # Add additional logs
    log_file = os.path.join(session.temp_dir, f"{session.current_video_id}_log.txt")
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                file_logs = f.readlines()
                # Only take the last 20 lines to avoid overwhelming the UI
                recent_logs = file_logs[-20:]
                logs.extend([log.strip() for log in recent_logs])
        except Exception as e:
            logs.append(f"Error reading logs: {str(e)}")
    
    # Add generic processing message if no specific logs
    if not logs and video_data.processing:
        logs.append(f"Processing video {session.current_video_id}... Please wait.")
    
    return {"logs": logs}

# New endpoints for dashboard

@router.get("/all", response_model=List[VideoSchema])
async def get_all_videos(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all videos for the current user to display in dashboard"""
    videos = db.query(Video).filter(Video.user_id == current_user.id).all()
    
    # Update alert counts to only include confirmed alerts
    for video in videos:
        confirmed_alert_count = db.query(VideoAlert).filter(
            VideoAlert.video_id == video.id,
            VideoAlert.is_confirmed_alert == True
        ).count()
        video.alert_count = confirmed_alert_count
    
    return videos

@router.get("/{video_id}/details", response_model=VideoDetail)
async def get_video_details(
    video_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific video"""
    video = db.query(Video).filter(
        Video.video_id == video_id,
        Video.user_id == current_user.id
    ).first()
    
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    
    # Update alert count to only include confirmed alerts
    confirmed_alert_count = db.query(VideoAlert).filter(
        VideoAlert.video_id == video.id,
        VideoAlert.is_confirmed_alert == True
    ).count()
    
    # Only return genuine suspicious alerts to the frontend
    video.alert_count = confirmed_alert_count
    
    # Add video_url to the response
    setattr(video, 'video_url', get_s3_url(video.video_s3_path) if video.video_s3_path else None)
    
    return video

@router.get("/{video_id}/logs")
async def get_video_logs(
    video_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get logs for a specific video"""
    # First check if the video exists and belongs to the user
    db_video = db.query(Video).filter(
        Video.video_id == video_id,
        Video.user_id == current_user.id
    ).first()
    
    if not db_video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    
    logs = []
    
    # Try to get logs from S3 if they exist
    if db_video.logs_s3_path:
        try:
            # Create a temporary file to download the logs
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                log_file_path = temp_file.name
            
            try:
                # The logs_s3_path contains the full S3 key
                s3.download_file(BUCKET_NAME, db_video.logs_s3_path, log_file_path)
                
                # Read the log file
                with open(log_file_path, 'r') as f:
                    log_lines = f.readlines()
                    logs = [line.strip() for line in log_lines if line.strip()]
            except Exception as e:
                logs.append(f"Error fetching logs: {str(e)}")
            finally:
                # Clean up the temporary file
                if os.path.exists(log_file_path):
                    os.remove(log_file_path)
        except Exception as e:
            logs.append(f"Error preparing for log download: {str(e)}")
    else:
        logs.append("No logs available for this video")
    
    return {"logs": logs}

@router.delete("/{video_id}")
async def delete_video(
    video_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a video and all its associated data"""
    # Check if the video exists and belongs to the user
    db_video = db.query(Video).filter(
        Video.video_id == video_id,
        Video.user_id == current_user.id
    ).first()
    
    if not db_video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    
    # Delete from S3 if base path exists
    if db_video.s3_base_path:
        try:
            # First, list all objects in the video's folder
            s3_prefix = db_video.s3_base_path
            response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=s3_prefix)
            
            # Delete all objects in the folder
            if 'Contents' in response:
                for obj in response['Contents']:
                    try:
                        s3.delete_object(Bucket=BUCKET_NAME, Key=obj['Key'])
                    except Exception as e:
                        print(f"Error deleting S3 object {obj['Key']}: {e}")
            
            print(f"Deleted S3 objects with prefix: {s3_prefix}")
        except Exception as e:
            print(f"Error deleting from S3: {e}")
    
    # Delete from memory if it exists
    if current_user.id in user_sessions:
        session = user_sessions[current_user.id]
        if video_id in session.videos:
            del session.videos[video_id]
            
            # If this was the current video, clear it
            if session.current_video_id == video_id:
                session.current_video_id = None
    
    # Delete from database (this will cascade to delete related logs and alerts)
    db.delete(db_video)
    db.commit()
    
    return {"message": f"Video {video_id} successfully deleted"} 