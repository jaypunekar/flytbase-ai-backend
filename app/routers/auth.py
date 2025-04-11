from fastapi import APIRouter, Depends, HTTPException, status, Form, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta, datetime
import logging

from ..database import get_db
from ..schemas.user import UserCreate, Token, User, VerifyEmail
from ..models.user import User as UserModel
from ..auth import authenticate_user, create_access_token, get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES
from ..utils.email import generate_verification_token, send_verification_email, is_token_valid

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
)

@router.post("/signup", response_model=dict)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    # Check if email already exists
    db_user = db.query(UserModel).filter(UserModel.email == user.email).first()
    if db_user:
        # If user exists but not verified, allow re-sending verification token
        if db_user and not db_user.is_verified:
            # Generate a new token and update the existing user
            verification_token = generate_verification_token()
            db_user.verification_token = verification_token
            db_user.token_created_at = datetime.now()
            db.commit()
            
            # Send verification email
            send_verification_email(db_user.email, verification_token)
            
            return {
                "message": "Verification email sent. Please verify your account.",
                "email": db_user.email,
                "requires_verification": True
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Check if passwords match
    if user.password != user.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match"
        )
    
    # Generate verification token
    verification_token = generate_verification_token()
    
    # Create new user (unverified)
    hashed_password = get_password_hash(user.password)
    db_user = UserModel(
        email=user.email, 
        hashed_password=hashed_password,
        is_active=True,
        is_verified=False,
        verification_token=verification_token,
        token_created_at=datetime.now()
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Send verification email
    send_success = send_verification_email(user.email, verification_token)
    
    if not send_success:
        # If email sending fails, still create the account but notify user
        return {
            "message": "Account created, but verification email could not be sent. Please contact support.",
            "email": db_user.email,
            "requires_verification": True
        }
    
    return {
        "message": "Verification email sent. Please verify your account.",
        "email": db_user.email,
        "requires_verification": True
    }

@router.post("/verify", response_model=Token)
def verify_email(verification_data: VerifyEmail, db: Session = Depends(get_db)):
    # Find user by email
    user = db.query(UserModel).filter(UserModel.email == verification_data.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if already verified
    if user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already verified"
        )
    
    # Check if token is valid
    if not is_token_valid(user.token_created_at):
        # Generate a new token and update the user
        verification_token = generate_verification_token()
        user.verification_token = verification_token
        user.token_created_at = datetime.now()
        db.commit()
        
        # Send verification email
        send_verification_email(user.email, verification_token)
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification token has expired. A new one has been sent to your email."
        )
    
    # Check if token matches
    if user.verification_token != verification_data.token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification token"
        )
    
    # Mark user as verified
    user.is_verified = True
    user.verification_token = None  # Clear the token once used
    db.commit()
    
    # Generate an access token for automatic login
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/resend-verification", response_model=dict)
def resend_verification(email: str = Form(...), db: Session = Depends(get_db)):
    # Find user by email
    user = db.query(UserModel).filter(UserModel.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if already verified
    if user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already verified"
        )
    
    # Generate a new token and update the user
    verification_token = generate_verification_token()
    user.verification_token = verification_token
    user.token_created_at = datetime.now()
    db.commit()
    
    # Send verification email
    send_success = send_verification_email(user.email, verification_token)
    
    if not send_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send verification email. Please try again later."
        )
    
    return {
        "message": "Verification email sent. Please verify your account.",
        "email": user.email
    }

# Login endpoint that accepts form data directly
@router.post("/login", response_model=Token)
async def login_with_form(
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, email, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is verified
    if not user.is_verified:
        # If user is not verified, return a specific error
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Please verify your email first.",
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"} 