# AI Video Analysis Platform - Simple Documentation

## Overview

This platform lets users upload or stream videos while an AI watches every frame to detect unusual activities in real time. The system also lets users chat with a bot to ask what's happening in the video or get a full summary.

## Features

- Safe user sign-up and login
- Upload video or stream live from phone camera
- Real-time detection of objects and suspicious activity
- Chatbot explains what is going on in the video
- Ask chatbot for a summary at any time
- Email alerts if any suspicious activity is found
- All past videos and chatbot conversations saved per user
- Unlimited video uploads and storage

# Installation Guide

To run the backend on your system:

1. Clone the repository:
```bash
https://github.com/jaypunekar/flytbase-ai-backend.git
```

2. Go into the folder:
```bash
cd flytbase-ai-backend
```

3. Create a virtual environment (Python version used: 3.12.8):
```bash
python -m venv venv
```

4. Activate the virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the backend server:
```bash
uvicorn app.main:app
```

Now the FastAPI backend will be running at `http://localhost:8000`.


## How I Built This

### Real-Time Understanding

I used **LangChain**, **RAG**, and a **Vision Language Model (VLM)** together. The VLM watches each video frame and tells what objects or actions it sees. LangChain and RAG take these frame-by-frame results and keep them in memory using vector embeddings. This allows the chatbot to explain what’s happening in the video, answer questions about earlier frames, and even give a full summary of the video while it's still running.

These embeddings are created every few frames to keep things fast and light. Once the video ends, everything is saved to AWS S3 so the user can view and chat with the same video/stream later.

### Real-Time Video Processing

As soon as the video is uploaded or streamed, it is broken into small frames using **OpenCV**. Each frame is then passed to the **Open VLM (Vision Language Model)** for object and activity detection. The VLM returns a description of what it sees—such as "a man walking," "a car speeding," or "a crowd gathering."

These descriptions are turned into embeddings and temporarily stored in memory. LangChain retrieves and uses these to answer questions, keep context, and generate summaries in real time. If anything looks suspicious, the backend triggers an email alert.

### Storage

I used **AWS S3** to save everything: videos, frame logs, vector data, and chatbot chat history. Each user’s data is stored separately.

### Scalability

The platform can handle many users and many videos at once because:

- FastAPI can be run on multiple servers with a load balancer
- AWS S3 gives us unlimited storage space
- AWS IVS handles live streams without delay

### Security

- Every user must log in to use the platform
- All videos and logs are private to the owner
- Data in AWS S3 is encrypted and access-controlled
- Chat logs, alerts, and metadata are stored securely

### Deployment

#### Frontend

The user interface is built in **React**. 

#### Backend

Backend logic is written in **FastAPI**. It controls file uploads, user logins, talks to the AI models, and handles chatbot responses.

#### AI Pipeline

- **OpenCV** breaks video into frames
- **Open VLM** checks each frame for objects, actions, and context
- **LangChain + RAG** uses temporary in-memory vector store to hold and retrieve frame insights
- **GPT-4o** powers the chatbot so it can explain or summarize what it sees

