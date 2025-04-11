import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import secrets
import string
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Email configuration
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Verification token constants
TOKEN_LENGTH = 6  # 6-digit token
TOKEN_EXPIRY_HOURS = 24  # Token valid for 24 hours

def generate_verification_token():
    """Generate a random 6-digit verification token"""
    # Generate a numeric token
    chars = string.digits
    return ''.join(secrets.choice(chars) for _ in range(TOKEN_LENGTH))

def is_token_valid(token_created_at):
    """Check if a token is still valid (not expired)"""
    if not token_created_at:
        return False
    
    expiry_time = token_created_at + timedelta(hours=TOKEN_EXPIRY_HOURS)
    return datetime.now() < expiry_time

def send_verification_email(to_email, token):
    """Send verification email with the token"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = to_email
        msg['Subject'] = "Verify Your FlytBase Video Analysis Account"
        
        # Create verification link
        verification_link = f"{FRONTEND_URL}/verify-email?email={to_email}&token={token}"
        
        # Email content
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eee; border-radius: 10px; background-color: #f9f9f9;">
                <h2 style="color: #1976d2;">Welcome to FlytBase Video Analysis!</h2>
                <p>Thank you for registering. Please verify your account using one of the methods below:</p>
                
                <div style="margin: 20px 0; text-align: center;">
                    <a href="{verification_link}" style="background-color: #1976d2; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; display: inline-block; font-weight: bold;">Verify Email Now</a>
                </div>
                
                <p>Or enter this verification code on the verification page:</p>
                <h1 style="color: #1976d2; text-align: center; font-size: 32px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; letter-spacing: 5px;">{token}</h1>
                
                <p>This code and link will expire in {TOKEN_EXPIRY_HOURS} hours.</p>
                <p>If you did not create an account, you can safely ignore this email.</p>
                <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
                <p style="font-size: 12px; color: #777;">Best regards,<br>FlytBase Team</p>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_FROM, to_email, text)
        server.quit()
        
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False

def send_alert_email(to_email, video_filename, alert_details, frame_id, timestamp, thumbnail_url=None):
    """Send email notification about suspicious activity detected in a video
    
    Args:
        to_email: User's email address
        video_filename: Name of the video where activity was detected
        alert_details: Description of the suspicious activity
        frame_id: Frame number where activity was detected
        timestamp: Timestamp in the video (seconds)
        thumbnail_url: Optional URL to the frame image showing the suspicious activity
    """
    if not to_email:
        print("Cannot send alert email: No recipient email provided")
        return False
        
    if not EMAIL_USERNAME or not EMAIL_PASSWORD:
        print("Cannot send alert email: Missing email credentials in environment variables")
        return False
    
    try:
        print(f"Preparing alert email to {to_email} for frame {frame_id}")
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = to_email
        msg['Subject'] = f"🚨 Security Alert: Suspicious Activity Detected in {video_filename}"
        
        # Format timestamp as minutes:seconds
        minutes = int(timestamp / 60)
        seconds = int(timestamp % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        # Dashboard link
        dashboard_link = f"{FRONTEND_URL}/dashboard"
        
        # Email content
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eee; border-radius: 10px; background-color: #f9f9f9;">
                <h2 style="color: #dc3545;">Security Alert: Suspicious Activity Detected</h2>
                
                <div style="background-color: #fff; border-left: 4px solid #dc3545; padding: 15px; margin: 15px 0; border-radius: 4px;">
                    <p><strong>Video:</strong> {video_filename}</p>
                    <p><strong>Frame:</strong> {frame_id}</p>
                    <p><strong>Timestamp:</strong> {time_str}</p>
                    <p><strong>Alert Type:</strong> Security Concern</p>
                </div>
                
                <h3>Detection Details:</h3>
                <p style="background-color: #fff; padding: 15px; border-radius: 4px;">{alert_details}</p>
                
                {f'<div style="margin: 20px 0; text-align: center;"><img src="{thumbnail_url}" style="max-width: 100%; border-radius: 4px; border: 1px solid #ddd;" alt="Suspicious activity" /></div>' if thumbnail_url else ''}
                
                <div style="margin: 20px 0; text-align: center;">
                    <a href="{dashboard_link}" style="background-color: #1976d2; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; display: inline-block; font-weight: bold;">Review in Dashboard</a>
                </div>
                
                <p>If you believe this is a false alarm, you can mark it as such in the dashboard.</p>
                
                <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
                <p style="font-size: 12px; color: #777;">This is an automated security alert from FlytBase Video Analysis.<br>© FlytBase, Inc.</p>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        print(f"Connecting to email server {EMAIL_HOST}:{EMAIL_PORT}")
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.set_debuglevel(1)  # Enable verbose debug output
        server.ehlo()  # Identify ourselves to the server
        print("Starting TLS connection")
        server.starttls()
        server.ehlo()  # Re-identify ourselves over TLS connection
        
        print(f"Logging in with username: {EMAIL_USERNAME}")
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        
        text = msg.as_string()
        print(f"Sending email from {EMAIL_FROM} to {to_email}")
        server.sendmail(EMAIL_FROM, to_email, text)
        print("Email sent successfully")
        
        server.quit()
        
        return True
    except smtplib.SMTPAuthenticationError as e:
        print(f"SMTP Authentication Error: {e}")
        return False
    except smtplib.SMTPException as e:
        print(f"SMTP Error: {e}")
        return False
    except Exception as e:
        print(f"Failed to send alert email: {str(e)}")
        return False 