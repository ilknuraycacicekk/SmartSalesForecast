import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

# Create engine
engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Dependency to get database session
def get_db():
    """
    Veritabanı bağlantı oturumu oluştur ve işlem tamamlandığında kapat.
    """
    db = SessionLocal()
    try:
        yield db
        # İşlem commit edilmemiş olabilir, bu durumda bile temizleme yapmalıyız
    except Exception:
        db.rollback()
        raise
    finally:
        db.close() 