from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from .database import engine, Base, run_migrations
from .models import user, video, stream  # Import models to register them with Base
from .routers import auth, video, stream

# Create tables in the database
Base.metadata.create_all(bind=engine)

# Run migrations after tables are created
try:
    run_migrations()
except Exception as e:
    print(f"Error running migrations: {e}")

app = FastAPI(
    title="FlytBase Video Analysis API",
    description="API for video analysis with JWT authentication",
    version="1.0.0",
)

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",  # React frontend
    "http://127.0.0.1:3000",
    "https://player.live-video.net",  # AWS IVS Player
    "https://*.live-video.net",      # AWS IVS endpoints 
    "https://*.ivs.amazonaws.com",   # AWS IVS API endpoints
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Content-Length", "Content-Range", "Content-Disposition"],
)

# Mount static files (for uploads and temporary files)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(auth.router)
app.include_router(video.router)
app.include_router(stream.router)

@app.get("/")
async def root():
    return {"message": "Welcome to FlytBase Video Analysis API"} 