import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, extract, text
from datetime import datetime
from ..db.models import Product, Category, Order, OrderDetail, Customer, Supplier


def get_products(db: Session, skip: int = 0, limit: int = 100):
    """Get all products from the database."""
    return db.query(Product).offset(skip).limit(limit).all()


def get_product(db: Session, product_id: int):
    """Get a specific product by ID."""
    return db.query(Product).filter(Product.product_id == product_id).first()


def get_sales_data(db: Session, start_date=None, end_date=None):
    """
    Extract sales data from the database.
    
    Returns a pandas DataFrame with order details including:
    - product_id, product_name
    - order_date
    - customer_id
    - quantity, unit_price, discount
    - calculated features (month, year, etc.)
    """
    # Use raw SQL query to ensure proper date handling
    query = """
    SELECT 
        od.product_id, 
        p.product_name,
        p.category_id,
        c.category_name,
        o.order_date,
        o.customer_id,
        od.quantity,
        od.unit_price,
        od.discount,
        p.supplier_id,
        s.company_name as supplier_name
    FROM 
        order_details od
    JOIN 
        products p ON od.product_id = p.product_id
    JOIN 
        orders o ON od.order_id = o.order_id
    JOIN 
        categories c ON p.category_id = c.category_id
    JOIN 
        suppliers s ON p.supplier_id = s.supplier_id
    WHERE 
        o.order_date IS NOT NULL
    """
    
    # Apply date filters if provided
    params = {}
    if start_date:
        query += " AND o.order_date >= :start_date"
        params['start_date'] = start_date
    if end_date:
        query += " AND o.order_date <= :end_date"
        params['end_date'] = end_date
    
    try:
        # Execute query and convert to DataFrame
        result = db.execute(text(query), params)
        columns = [
            'product_id', 'product_name', 'category_id', 'category_name',
            'order_date', 'customer_id', 'quantity', 'unit_price', 'discount',
            'supplier_id', 'supplier_name'
        ]
        df = pd.DataFrame(result.fetchall(), columns=columns)
        
        if df.empty:
            print("Veri tabanından veri çekilemedi (boş DataFrame)")
            return df
        
        print(f"Çekilen ham veri: {len(df)} satır")
        
        # Sayısal ve kategorik sütunları tanımla
        numeric_cols = ['quantity', 'unit_price', 'discount']
        cat_cols = ['product_id', 'category_id', 'customer_id', 'supplier_id']
        
        # Kategorik değişkenleri string'e dönüştür
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Eksik ve geçersiz değerleri kontrol et ve düzelt
        for col in numeric_cols:
            if col in df.columns:
                # Eksik değerleri doldur
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())
                
                # Negatif veya sıfır değerleri düzelt
                if (df[col] <= 0).any():
                    df.loc[df[col] <= 0, col] = df[col].median()
                
                # Aykırı değerleri tespit et ve düzelt
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    outliers = abs(df[col] - mean) > 3 * std
                    if outliers.sum() > 0:
                        print(f"{col} sütununda {outliers.sum()} aykırı değer tespit edildi")
                        df.loc[outliers, col] = df[col].median()
        
        # Kategorik sütunlardaki eksik değerleri doldur
        for col in cat_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Tarih işlemleri
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'])
            min_date = datetime(1990, 1, 1)
            max_date = datetime.now()
            
            invalid_dates = (df['order_date'] < min_date) | (df['order_date'] > max_date)
            if invalid_dates.any():
                valid_dates = df.loc[~invalid_dates, 'order_date']
                df.loc[invalid_dates, 'order_date'] = valid_dates.median()
            
            # Sadece gerekli tarih özelliklerini ekle
            df['year'] = df['order_date'].dt.year
            df['month'] = df['order_date'].dt.month
        
        # Gelir hesapla
        if all(col in df.columns for col in ['quantity', 'unit_price', 'discount']):
            df['revenue'] = df['quantity'] * df['unit_price'] * (1 - df['discount'])
        
        print(f"Veri temizliği sonrası: {len(df)} satır, {len(df.columns)} sütun")
        return df
        
    except Exception as e:
        print(f"Veri çekme ve temizleme işlemi sırasında hata: {e}")
        return pd.DataFrame()


def get_monthly_sales_summary(db: Session, start_date=None, end_date=None):
    """
    Get monthly sales summary for all products.
    Returns a DataFrame with product_id, month, year, total_quantity, total_revenue
    """
    sales_df = get_sales_data(db, start_date, end_date)
    
    if sales_df.empty:
        return pd.DataFrame()
    
    # Kategorik değişkenleri sayısal değerlere dönüştür
    sales_df['category_id'] = sales_df['category_id'].astype(str)
    sales_df['supplier_id'] = sales_df['supplier_id'].astype(str)
    
    # Metin sütunlarını kaldır
    text_columns = ['product_name', 'category_name', 'supplier_name']
    sales_df = sales_df.drop(columns=[col for col in text_columns if col in sales_df.columns])
    
    summary = sales_df.groupby(['product_id', 'category_id', 'supplier_id', 'year', 'month']).agg({
        'quantity': 'sum',
        'revenue': 'sum',
        'unit_price': 'mean'
    }).reset_index()
    
    summary.rename(columns={
        'quantity': 'total_quantity', 
        'revenue': 'total_revenue',
        'unit_price': 'avg_price'
    }, inplace=True)
    
    return summary


def get_product_category_summary(db: Session):
    """
    Get sales summary grouped by product category.
    """
    query = text("""
    SELECT 
        c.category_id, 
        c.category_name, 
        COUNT(DISTINCT p.product_id) as product_count,
        SUM(od.quantity) as total_quantity,
        SUM(od.quantity * od.unit_price * (1-od.discount)) as total_revenue
    FROM 
        categories c
    JOIN 
        products p ON c.category_id = p.category_id
    JOIN 
        order_details od ON p.product_id = od.product_id
    JOIN 
        orders o ON od.order_id = o.order_id
    GROUP BY 
        c.category_id, c.category_name
    ORDER BY 
        total_revenue DESC
    """)
    
    result = db.execute(query)
    columns = ['category_id', 'category_name', 'product_count', 'total_quantity', 'total_revenue']
    return pd.DataFrame(result.fetchall(), columns=columns)


def prepare_training_data(db: Session):
    """
    Satış tahmin modeli için eğitim verilerini hazırla.
    X (özellikler) ve y (hedef) DataFrame'lerini döndürür.
    """
    monthly_data = get_monthly_sales_summary(db)
    
    if monthly_data.empty:
        return None, None
    
    print(f"Hazırlanan veri setinde {len(monthly_data)} satır var.")
    
    # Metin sütunlarını kaldır
    text_columns = ['product_name', 'category_name', 'supplier_name']
    monthly_data = monthly_data.drop(columns=[col for col in text_columns if col in monthly_data.columns])
    
    # Eksik verileri doldur
    for col in ['category_id', 'supplier_id']:
        if col in monthly_data.columns and monthly_data[col].isnull().sum() > 0:
            monthly_data[col] = monthly_data[col].fillna(monthly_data[col].mode()[0])
    
    # Sayısal sütunlardaki eksik değerleri doldur
    for col in ['total_quantity', 'total_revenue', 'avg_price']:
        if col in monthly_data.columns and monthly_data[col].isnull().sum() > 0:
            monthly_data[col] = monthly_data[col].fillna(monthly_data[col].median())
    
    # Aykırı değerleri kontrol et ve düzelt
    for col in ['total_quantity', 'total_revenue', 'avg_price']:
        if col in monthly_data.columns:
            mean = monthly_data[col].mean()
            std = monthly_data[col].std()
            if std > 0:
                outliers = abs(monthly_data[col] - mean) > 3 * std
                if outliers.sum() > 0:
                    print(f"{col} sütununda {outliers.sum()} aykırı değer tespit edildi.")
                    monthly_data.loc[outliers, col] = monthly_data[col].median()
    
    # Kategorik değişkenleri one-hot encoding ile dönüştür
    categorical_cols = ['category_id', 'supplier_id']
    for col in categorical_cols:
        if col in monthly_data.columns:
            # Kategorik değişkeni string'e dönüştür
            monthly_data[col] = monthly_data[col].astype(str)
            dummies = pd.get_dummies(monthly_data[col], prefix=col)
            monthly_data = pd.concat([monthly_data, dummies], axis=1)
            # Orijinal kategorik sütunu kaldır
            monthly_data = monthly_data.drop(columns=[col])
    
    # Özellikler ve hedef değişkeni ayır
    feature_cols = [
        'product_id', 'year', 'month',
        'total_revenue', 'avg_price'
    ] + [col for col in monthly_data.columns if col.startswith(('category_', 'supplier_'))]
    
    X = monthly_data[feature_cols]
    y = monthly_data['total_quantity']
    
    print(f"Veri temizliği sonrasında {len(X)} satır ve {len(X.columns)} sütun var.")
    print("Veri kalitesi özeti:")
    print(f"Toplam örnek sayısı: {len(X)}")
    print(f"Özellik sayısı: {len(X.columns)}")
    
    return X, y


def prepare_prediction_features(db: Session, product_id: int, order_date: datetime, customer_id=None, quantity=None):
    """
    Tahmin için özellik verilerini hazırla.
    Tahmin için gereken özellikleri içeren tek satırlık bir DataFrame döndürür.
    
    Args:
        db: Veritabanı bağlantısı
        product_id: Ürün ID
        order_date: Sipariş tarihi
        customer_id: Müşteri ID (opsiyonel)
        quantity: Tahmin edilecek miktar (opsiyonel)
    """
    try:
        # Ürün bilgisini al
        product = get_product(db, product_id)
        if not product:
            return None
        
        # Tüm kategorileri ve tedarikçileri bir kerede al
        categories = {c.category_id for c in db.query(Category.category_id).all()}
        suppliers = {s.supplier_id for s in db.query(Supplier.supplier_id).all()}
        
        # Ürün için ortalama satış değerlerini al (varsa)
        avg_quantity = None
        avg_revenue = None
        try:
            # Son 10 siparişteki ortalama miktar ve geliri bulalım (ORDER BY kısmını kaldırarak)
            query = text("""
            SELECT AVG(od.quantity) as avg_quantity, 
                   AVG(od.quantity * od.unit_price * (1-od.discount)) as avg_revenue
            FROM order_details od
            JOIN orders o ON od.order_id = o.order_id
            WHERE od.product_id = :product_id
            LIMIT 10
            """)
            result = db.execute(query, {"product_id": product_id})
            row = result.fetchone()
            if row:
                avg_quantity = row[0] or 0
                avg_revenue = row[1] or 0
                
            # Eğer transaction'da bir sorun olduysa, rollback yapıp temiz hale getirelim
            db.commit()
        except Exception as e:
            print(f"Satış verisi alınırken hata: {e}")
            db.rollback()  # Hata durumunda transaction'ı geri al
            # Hata olursa varsayılan değerleri kullanacağız
        
        # Değerleri varsayılanlara ayarla
        if avg_quantity is None:
            avg_quantity = 10  # Varsayılan değer
        if avg_revenue is None:
            avg_revenue = product.unit_price * 10  # Varsayılan değer
        
        # Eğer quantity parametresi verilmişse, geliri hesaplamak için kullan
        if quantity is not None:
            total_revenue = product.unit_price * quantity
        else:
            total_revenue = avg_revenue
        
        # Özellik değerlerini hazırla
        features = {
            'product_id': product_id,
            'year': order_date.year,
            'month': order_date.month,
            'avg_price': product.unit_price,
            'total_revenue': total_revenue,
            'category_id': product.category_id if product.category_id else 0,
            'supplier_id': product.supplier_id if product.supplier_id else 0,
        }
        
        # Kategori one-hot encoding
        for cat_id in categories:
            features[f'category_{cat_id}'] = 1 if cat_id == product.category_id else 0
        
        # Tedarikçi one-hot encoding
        for sup_id in suppliers:
            features[f'supplier_{sup_id}'] = 1 if sup_id == product.supplier_id else 0
        
        # Müşteri bilgisi varsa ve aktif bir transaction yoksa, müşteriyle ilgili özellikleri ekleyelim
        if customer_id:
            try:
                # Yeni bir session kullanmak yerine, mevcut session'ı temizleyip kullan
                db.rollback()  # Olası hataları temizlemek için
                customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
                if customer and hasattr(customer, 'country'):
                    # Örnek olarak country bilgisini ekleyelim
                    features['customer_country'] = hash(customer.country) % 10  # Basit bir numerik değer
                db.commit()  # Başarılı sorguyu commit et
            except Exception as e:
                print(f"Müşteri bilgisi alınırken hata: {e}")
                db.rollback()  # Hata durumunda transaction'ı geri al
                # Hata olsa bile devam et, bu önemli bir özellik değil
        
        # DataFrame'e dönüştür
        X = pd.DataFrame([features])
        
        return X
        
    except Exception as e:
        # Genel hata durumunda
        print(f"Tahmin özellikleri hazırlanırken hata: {e}")
        db.rollback()  # Transaction'ı geri al
        return None


def get_top_selling_products(db: Session, limit=10):
    """
    Get top selling products by quantity.
    """
    query = text("""
    SELECT 
        p.product_id, 
        p.product_name, 
        c.category_name,
        SUM(od.quantity) as total_quantity,
        SUM(od.quantity * od.unit_price * (1-od.discount)) as total_revenue
    FROM 
        products p
    JOIN 
        order_details od ON p.product_id = od.product_id
    JOIN 
        categories c ON p.category_id = c.category_id
    GROUP BY 
        p.product_id, p.product_name, c.category_name
    ORDER BY 
        total_quantity DESC
    LIMIT :limit
    """)
    
    result = db.execute(query, {"limit": limit})
    columns = ['product_id', 'product_name', 'category_name', 'total_quantity', 'total_revenue']
    df = pd.DataFrame(result.fetchall(), columns=columns)
    
    return df 