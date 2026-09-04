"""
Database Connection and Session Management for Vehicle Fault Classifier
Connects to Supabase PostgreSQL with connection pooling and graceful SQLite fallback.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # In case someone passes raw unencoded password with '@'
    if "@" in DATABASE_URL.split("://")[-1].split("@aws")[0] and "%40" not in DATABASE_URL:
        # URL encode '@' in password if necessary
        parts = DATABASE_URL.split("://")
        proto = parts[0]
        rest = parts[1]
        if ":" in rest and "@" in rest:
            user_pass, host_db = rest.rsplit("@", 1)
            if ":" in user_pass:
                user, pwd = user_pass.split(":", 1)
                pwd_encoded = pwd.replace("@", "%40")
                DATABASE_URL = f"{proto}://{user}:{pwd_encoded}@{host_db}"

    try:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=5,
            max_overflow=10,
        )
        # Quick connectivity test
        with engine.connect() as conn:
            pass
        print("[DB] Connected successfully to primary PostgreSQL database.")
    except Exception as e:
        print(f"[DB WARNING] Failed to connect to primary DB: {e}. Falling back to SQLite.")
        DATABASE_URL = "sqlite:///./vfc_history.db"
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    print("[DB INFO] DATABASE_URL not set in environment or .env. Using SQLite fallback.")
    DATABASE_URL = "sqlite:///./vfc_history.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency for obtaining a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
