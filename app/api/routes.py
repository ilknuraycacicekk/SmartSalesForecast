from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from ..db.database import get_db
from ..db.models import Product
from ..schemas.schemas import (
    Product as ProductSchema,
    SalesSummary,
    PredictionRequest,
    PredictionResponse,
    SalesPredictionMetrics,
    RetrainRequest,
    CategorySummary,
    TopSellingProduct
)
from ..services.data_service import (
    get_products,
    get_product,
    get_monthly_sales_summary,
    prepare_prediction_features,
    get_product_category_summary,
    get_top_selling_products
)
from ..services.ml_service import SalesForecastModel

router = APIRouter()
model = SalesForecastModel()

@router.get("/products", response_model=List[ProductSchema])
def read_products(
    skip: int = 0, 
    limit: int = 100, 
    category_id: Optional[int] = None,
    discontinued: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """
    Ürün listesini döndürür.
    
    Bu endpoint ile veritabanındaki ürünlerin listesini alabilirsiniz. 
    Kategori ve duruma göre filtreleme yapabilirsiniz.
    
    Args:
        skip: Atlanacak kayıt sayısı (sayfalama için)
        limit: Maksimum kayıt sayısı (sayfalama için)
        category_id: Filtreleme için kategori ID
        discontinued: Üretimi devam eden/etmeyen ürünleri filtreleme
        
    Returns:
        Ürün listesi
    """
    query = db.query(Product)
    
    if category_id is not None:
        query = query.filter(Product.category_id == category_id)
        
    if discontinued is not None:
        query = query.filter(Product.discontinued == (1 if discontinued else 0))
    
    products = query.offset(skip).limit(limit).all()
    return products


@router.get("/products/{product_id}", response_model=ProductSchema)
def read_product(product_id: int, db: Session = Depends(get_db)):
    """
    Get a specific product by ID.
    """
    product = get_product(db, product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID {product_id} not found"
        )
    return product


@router.get("/sales_summary", response_model=List[SalesSummary])
def read_sales_summary(
    year: Optional[int] = None,
    month: Optional[int] = None,
    product_id: Optional[int] = None,
    category_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Satış özet verilerini döndürür.
    
    Bu endpoint ile aylık satış özeti verilerini alabilirsiniz.
    Tarih, ürün ve kategori bazında filtreleme yapabilirsiniz.
    
    Args:
        year: Filtreleme için yıl
        month: Filtreleme için ay
        product_id: Filtreleme için ürün ID
        category_id: Filtreleme için kategori ID
        
    Returns:
        Satış özeti verileri listesi
    """
    # Get date filters
    start_date = None
    end_date = None
    
    if year is not None:
        if month is not None:
            # Filter by specific month
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        else:
            # Filter by year
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
    
    # Get sales summary
    summary = get_monthly_sales_summary(db, start_date, end_date)
    
    if summary.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Satış verisi bulunamadı"
        )
    
    # Apply additional filters
    if product_id is not None:
        summary = summary[summary['product_id'] == product_id]
        
    if category_id is not None:
        summary = summary[summary['category_id'] == category_id]
    
    if summary.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Belirtilen filtrelerle satış verisi bulunamadı"
        )
    
    return summary.to_dict('records')


@router.get("/category_summary", response_model=List[CategorySummary])
def read_category_summary(db: Session = Depends(get_db)):
    """
    Kategori bazlı satış özeti döndürür.
    
    Bu endpoint ile ürün kategorilerine göre gruplandırılmış satış verilerini alabilirsiniz.
    Her kategorinin toplam satış miktarı ve geliri listelenir.
    
    Returns:
        Kategori bazlı satış özeti listesi
    """
    summary = get_product_category_summary(db)
    
    if summary.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kategori verisi bulunamadı"
        )
    
    return summary.to_dict('records')


@router.get("/top_products", response_model=List[TopSellingProduct])
def read_top_products(limit: int = Query(10, ge=1, le=50), db: Session = Depends(get_db)):
    """
    En çok satan ürünleri döndürür.
    
    Bu endpoint ile en çok satan ürünlerin listesini alabilirsiniz.
    Satış miktarına göre sıralanmış şekilde sonuçları gösterir.
    
    Args:
        limit: Listelenecek maksimum ürün sayısı (1-50 arası)
        
    Returns:
        En çok satan ürünlerin listesi
    """
    products = get_top_selling_products(db, limit)
    
    if products.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ürün satış verisi bulunamadı"
        )
    
    return products.to_dict('records')


@router.post("/predict", response_model=PredictionResponse)
def predict_sales(
    prediction_request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Satış tahmini yapar.
    
    Bu endpoint ürün, tarih ve müşteri bilgileri kullanarak gelecek satış miktarı tahmini yapar.
    
    Args:
        product_id: Tahmin yapılacak ürünün ID'si
        order_date: Tahmin için tarih
        customer_id: Müşteri ID (opsiyonel)
        quantity: Başlangıç miktarı (opsiyonel)
    
    Returns:
        product_id: Ürün ID
        product_name: Ürün adı
        predicted_quantity: Tahmini satış miktarı
        confidence: Tahmin güven oranı
        timestamp: Tahmin zamanı
    """
    try:
        # Model yüklü değilse, yükle
        if model.model is None:
            success = model.load()
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model henüz eğitilmemiş. Lütfen önce /retrain endpoint'ini kullanarak modeli eğitin."
                )
        
        # Ürünün var olup olmadığını kontrol et
        product = get_product(db, prediction_request.product_id)
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ürün ID {prediction_request.product_id} bulunamadı"
            )
        
        # Her işlem öncesi transaction'ı temizle
        db.rollback()
        
        # Tahmin için özellikleri hazırla
        X = prepare_prediction_features(
            db,
            prediction_request.product_id,
            prediction_request.order_date,
            prediction_request.customer_id,
            prediction_request.quantity
        )
        
        if X is None:
            # Hata durumunda transaction'ı temizle
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tahmin için özellikler hazırlanamadı"
            )
        
        # Tahmini yap
        prediction = model.predict(X)
        
        if prediction is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Tahmin yaparken hata oluştu"
            )
        
        # Güveni hesapla (basit yaklaşım - geliştirilebilir)
        confidence = 0.8
        if model.metrics and 'r2_score' in model.metrics:
            # R² değerini güven göstergesi olarak kullan
            confidence = max(0.5, min(0.95, model.metrics['r2_score']))
        
        # İşlem başarılı, commit yap
        db.commit()
        
        # Tahmin edilen değeri pozitif yap ve yuvarlama ile daha anlamlı hale getir
        predicted_value = max(0, float(prediction[0]))
        rounded_prediction = round(predicted_value)  # Tam sayıya yuvarla
        
        # Yanıtı hazırla
        return PredictionResponse(
            product_id=prediction_request.product_id,
            product_name=product.product_name,
            predicted_quantity=rounded_prediction,
            confidence=confidence,
            timestamp=datetime.now()
        )
    
    except HTTPException:
        # HTTPException'ı tekrar fırlat
        raise
    
    except Exception as e:
        # Beklenmeyen bir hata olursa transaction'ı geri al
        db.rollback()
        print(f"Tahmin işleminde beklenmeyen hata: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tahmin işleminde beklenmeyen bir hata oluştu: {str(e)}"
        )


@router.post("/retrain", response_model=SalesPredictionMetrics)
def retrain_model(
    retrain_request: RetrainRequest = None,
    db: Session = Depends(get_db)
):
    """
    Satış tahmin modelini yeniden eğitir.
    
    Bu endpoint ile makine öğrenmesi modelini yeniden eğitebilirsiniz.
    
    Args:
        model_type: Eğitilecek model tipi ('decision_tree', 'linear', 'knn', 'logistic')
        
    Returns:
        r2_score: R-kare değeri (modelin açıklayıcılık gücü)
        rmse: Root Mean Squared Error (hata karesi ortalamasının karekökü)
        mae: Mean Absolute Error (ortalama mutlak hata)
        accuracy: Doğruluk değeri
        threshold: Doğruluk hesaplaması için kullanılan eşik değeri
        model_info: Model hakkında diğer bilgiler
    """
    try:
        # Varsayılan parametreleri ayarla
        model_type = "decision_tree"
        
        # İstekten parametreleri güncelle
        if retrain_request and retrain_request.model_type:
            model_type = retrain_request.model_type
        
        # Model tipinin geçerli olup olmadığını kontrol et
        valid_models = ["decision_tree", "linear", "knn", "logistic"]
        if model_type not in valid_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Geçersiz model tipi. Şunlardan biri olmalı: {', '.join(valid_models)}"
            )
        
        # Modeli eğit
        metrics = model.train(
            db, 
            model_type=model_type
        )
        
        if "error" in metrics:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=metrics["error"]
            )
        
        # Model bilgisini al
        model_info = model.get_model_info()
        
        # Metrikleri ve model bilgisini döndür
        return SalesPredictionMetrics(
            r2_score=metrics["r2_score"],
            rmse=metrics["rmse"],
            mae=metrics["mae"],
            accuracy=metrics["accuracy"],
            threshold=metrics["threshold"],
            model_info=model_info
        )
        
    except HTTPException:
        # HTTPException'ı tekrar fırlat
        raise
    
    except Exception as e:
        # Beklenmeyen bir hata olursa
        print(f"Model eğitim işleminde beklenmeyen hata: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model eğitiminde beklenmeyen bir hata oluştu: {str(e)}"
        )


@router.get("/model_info", response_model=SalesPredictionMetrics)
def get_model_info(db: Session = Depends(get_db)):
    """
    Get information about the trained model.
    """
    # Load model if not already loaded
    if model.model is None:
        success = model.load()
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not trained yet"
            )
    
    # Get model info
    model_info = model.get_model_info()
    
    # Return metrics and model info
    return SalesPredictionMetrics(
        r2_score=model_info["metrics"]["r2_score"],
        rmse=model_info["metrics"]["rmse"],
        mae=model_info["metrics"]["mae"],
        accuracy=model_info["metrics"]["accuracy"] if "accuracy" in model_info["metrics"] else 0.0,
        threshold=model_info["metrics"].get("threshold"),
        model_info=model_info
    ) 