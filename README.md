# Smart Sales Forecast API

A machine learning-based sales forecast API built with FastAPI, using Northwind database data to predict product sales.

## Features

- Sales data analysis from Northwind database
- Machine learning model for sales prediction
- REST API endpoints for products, predictions, and sales summaries
- Optional model retraining capability

## Technical Stack

- **Backend**: FastAPI
- **Database**: PostgreSQL (Northwind)
- **ORM**: SQLAlchemy
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn
- **Documentation**: Swagger (via FastAPI)

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure database connection in `.env` file
4. Run the application:
   ```
   uvicorn app.main:app --reload
   ```

## API Endpoints

- GET `/products`: List all products
- GET `/sales_summary`: Get sales summary data
- POST `/predict`: Get sales prediction for a product
- POST `/retrain`: Retrain the prediction model (optional)

## Documentation

API documentation is available at `/docs` after starting the server. 


