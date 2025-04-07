from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .api.routes import router as api_router
from .db.database import engine, Base, get_db

# Create FastAPI app
app = FastAPI(
    title="Sales Forecast API",
    description="API for predicting product sales based on historical Northwind data",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(
    api_router,
    prefix="/api",
    tags=["api"],
)

# Create database tables
Base.metadata.create_all(bind=engine)

@app.get("/")
def root():
    """
    Root endpoint with API info.
    """
    return {
        "message": "Welcome to Sales Forecast API",
        "docs": "/docs",
        "version": "1.0.0",
    }

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint.
    """
    try:
        # Check database connection
        result = db.execute("SELECT 1").first()
        db_healthy = result is not None
    except Exception:
        db_healthy = False

    return {
        "status": "healthy" if db_healthy else "unhealthy",
        "database": "connected" if db_healthy else "disconnected",
    } 