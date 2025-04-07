from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ProductBase(BaseModel):
    product_name: str
    unit_price: float
    category_id: Optional[int] = None
    supplier_id: Optional[int] = None


class ProductCreate(ProductBase):
    quantity_per_unit: Optional[str] = None
    units_in_stock: Optional[int] = None
    units_on_order: Optional[int] = None
    reorder_level: Optional[int] = None
    discontinued: int = 0


class Product(ProductBase):
    product_id: int
    quantity_per_unit: Optional[str] = None
    units_in_stock: Optional[int] = None
    units_on_order: Optional[int] = None
    reorder_level: Optional[int] = None
    discontinued: int

    class Config:
        from_attributes = True


class CategoryBase(BaseModel):
    category_name: str
    description: Optional[str] = None


class Category(CategoryBase):
    category_id: int

    class Config:
        from_attributes = True


class SupplierBase(BaseModel):
    company_name: str
    contact_name: Optional[str] = None
    contact_title: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    phone: Optional[str] = None
    fax: Optional[str] = None


class Supplier(SupplierBase):
    supplier_id: int

    class Config:
        from_attributes = True


class SalesSummary(BaseModel):
    product_id: int
    product_name: str
    total_quantity: int
    total_revenue: float
    avg_price: float
    year: int
    month: int
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    supplier_id: Optional[int] = None
    supplier_name: Optional[str] = None

    class Config:
        from_attributes = True


class CategorySummary(BaseModel):
    category_id: int
    category_name: str
    product_count: int
    total_quantity: int
    total_revenue: float

    class Config:
        from_attributes = True


class TopSellingProduct(BaseModel):
    product_id: int
    product_name: str
    category_name: str
    total_quantity: int
    total_revenue: float

    class Config:
        from_attributes = True


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class PredictionRequest(BaseModel):
    product_id: int = Field(..., description="Tahmin yapılacak ürünün ID'si")
    order_date: datetime = Field(..., description="Tahmin için kullanılacak tarih")
    customer_id: Optional[str] = Field(None, description="Opsiyonel: Müşteri ID")
    quantity: Optional[float] = Field(None, description="Opsiyonel: Tahmin edilen başlangıç miktarı")
    features: Optional[Dict[str, Any]] = Field({}, description="Opsiyonel: Ek tahmin özellikleri")
    
    class Config:
        schema_extra = {
            "example": {
                "product_id": 5,
                "order_date": "2023-06-01T00:00:00",
                "customer_id": "VINET",
                "quantity": 10,
                "features": {}
            }
        }


class PredictionResponse(BaseModel):
    product_id: int = Field(..., description="Tahmin yapılan ürünün ID'si")
    product_name: str = Field(..., description="Ürün adı")
    predicted_quantity: float = Field(..., description="Tahmini satış miktarı")
    confidence: float = Field(..., description="Tahmin güven oranı (0-1 arası)")
    timestamp: datetime = Field(..., description="Tahmin yapılma zamanı")
    
    class Config:
        schema_extra = {
            "example": {
                "product_id": 5,
                "product_name": "Chef Anton's Gumbo Mix",
                "predicted_quantity": 12,
                "confidence": 0.89,
                "timestamp": "2023-06-01T12:30:45.123456"
            }
        }


class SalesPredictionMetrics(BaseModel):
    r2_score: float
    rmse: float
    mae: float
    accuracy: float
    threshold: Optional[float] = None
    model_info: Dict[str, Any]


class RetrainRequest(BaseModel):
    model_type: Optional[str] = Field(
        None, 
        description="Model tipi: 'decision_tree', 'linear', 'knn', 'logistic'",
        example="decision_tree"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "decision_tree"
            }
        } 