from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment or use SQLite as fallback
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Fallback to SQLite
    DATABASE_URL = "sqlite:///./app.db"
    print(f"DATABASE_URL not found in environment. Using SQLite: {DATABASE_URL}")

# Create engine with appropriate settings
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Add migration functionality
def add_column_if_not_exists(engine, table_name, column_name, column_type):
    """Add a column to a table if it doesn't exist"""
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    
    if column_name not in columns:
        if engine.dialect.name == "sqlite":
            # SQLite doesn't support ALTER TABLE ADD COLUMN with standard syntax
            # We need a different approach for SQLite
            try:
                engine.execute(text(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};'))
                print(f"Added column {column_name} to {table_name}")
            except Exception as e:
                print(f"Could not add column {column_name} to {table_name}: {e}")
                print("You may need to manually add this column if using SQLite")
        else:
            engine.execute(text(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};'))
            print(f"Added column {column_name} to {table_name}")
    else:
        print(f"Column {column_name} already exists in {table_name}")

def run_migrations():
    """Run all database migrations"""
    # Add is_confirmed_alert column to video_alerts table if it doesn't exist
    add_column_if_not_exists(engine, 'video_alerts', 'is_confirmed_alert', 'BOOLEAN DEFAULT TRUE')
    
    print("Database migrations completed successfully") 