import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
from dotenv import load_dotenv
from .data_service import prepare_training_data, get_monthly_sales_summary

# Load environment variables
load_dotenv()

# Get model path from environment
MODEL_PATH = os.getenv("MODEL_PATH", "app/models/sales_forecast_model.pkl")

class SalesForecastModel:
    """
    Satış tahmini yapan makine öğrenmesi modeli.
    """
    
    def __init__(self):
        """Model sınıfını başlat."""
        self.model = None
        self.features = None
        self.metrics = None
        self.trained_date = None
        self.model_type = "decision_tree"
    
    def train(self, db, model_type="decision_tree", test_size=0.2, random_state=42):
        """
        Satış tahmin modelini eğit.
        
        Args:
            db: Veritabanı bağlantısı
            model_type: Model tipi ('decision_tree', 'linear', 'knn', 'logistic')
            test_size: Test verisi oranı
            random_state: Rastgele sayı üreteci için sabit değer
            
        Returns:
            dict: Model metrikleri
        """
        # Model tipini kaydet
        self.model_type = model_type
        
        print(f"\n-------- {model_type.upper()} MODELİ EĞİTİLİYOR --------")
        print("Veri hazırlanıyor ve temizleniyor...")
        
        # Eğitim verilerini hazırla (bu aşamada eksik veri kontrolü ve temizliği de yapılır)
        X, y = prepare_training_data(db)
        
        if X is None or y is None or len(X) < 10:
            return {"error": "Eğitim için yeterli veri yok"}
        
        # Özellik isimlerini kaydet
        self.features = X.columns.tolist()
        
        # Veri kalitesi raporu
        print("\nVeri kalitesi özeti:")
        print(f"Toplam örnek sayısı: {len(X)}")
        print(f"Özellik sayısı: {len(self.features)}")
        
        # Eğitim ve test verilerini ayır
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Eğitim seti: {len(X_train)} örnek")
        print(f"Test seti: {len(X_test)} örnek")
        
        # Model tipine göre model oluştur
        if model_type == "decision_tree":
            model = DecisionTreeRegressor(random_state=random_state)
        elif model_type == "linear":
            model = LinearRegression()
        elif model_type == "knn":
            model = KNeighborsRegressor(n_neighbors=5)
        elif model_type == "logistic":
            model = LogisticRegression(random_state=random_state)
        else:
            return {"error": f"Bilinmeyen model tipi: {model_type}"}
        
        # Pipeline oluştur
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Verileri ölçeklendir
            ('regressor', model)           # Modeli eğit
        ])
        
        # Modeli eğit
        self.model = pipeline
        self.model.fit(X_train, y_train)
        
        # Modeli değerlendir
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Doğruluk hesapla (tahminlerin belirli bir eşik değerin altında olma oranı)
        threshold = 0.2 * np.mean(np.abs(y_test))  # Eşik değeri olarak ortalama hedefin %20'si
        accuracy = np.mean(np.abs(y_test - y_pred) < threshold)
        
        # Metrikleri kaydet
        self.metrics = {
            "r2_score": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "accuracy": float(accuracy),
            "threshold": float(threshold),
            "n_samples": len(X),
            "test_size": test_size,
            "model_type": model_type
        }
        
        # Eğitim tarihini güncelle
        self.trained_date = datetime.now()
        
        # Modeli kaydet
        self.save()
        
        return self.metrics
    
    def predict(self, X):
        """
        Satış tahmini yap.
        
        Args:
            X: Tahmin için özellikler içeren DataFrame
            
        Returns:
            array: Tahmin edilen satış miktarları
        """
        if self.model is None:
            self.load()
            
        if self.model is None:
            return None
        
        # Sütun sırasını eğitim verisindeki sıraya göre ayarla
        if self.features is not None:
            # Eğitim sırasında kullanılmayan özellikleri tahmin DataFrame'inden çıkar
            X = X[[col for col in X.columns if col in self.features]]
            
            # Eksik özellikleri 0 değeriyle doldur
            for feature in self.features:
                if feature not in X.columns:
                    X[feature] = 0
            
            # Özellikleri aynı sıraya getir
            X = X[self.features]
        
        # Tahmin yap
        predictions = self.model.predict(X)
        
        return predictions
    
    def get_feature_importance(self):
        """
        Decision Tree modeli için özellik önemlerini al.
        Sadece decision_tree modeli için çalışır.
        
        Returns:
            DataFrame: Özellikler ve önem değerleri
        """
        if self.model is None or self.model_type != "decision_tree":
            return None
            
        # Özellik önemlerini çıkar
        model = self.model.named_steps['regressor']
        importance = model.feature_importances_
        
        # DataFrame oluştur
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': importance
        })
        
        # Öneme göre sırala
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save(self):
        """Modeli dosyaya kaydet."""
        # Dizin yoksa oluştur
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Modeli, özellikleri, metrikleri ve eğitim tarihini kaydet
        joblib.dump({
            'model': self.model,
            'features': self.features,
            'metrics': self.metrics,
            'trained_date': self.trained_date,
            'model_type': self.model_type
        }, MODEL_PATH)
    
    def load(self):
        """Modeli dosyadan yükle."""
        try:
            if os.path.exists(MODEL_PATH):
                data = joblib.load(MODEL_PATH)
                self.model = data['model']
                self.features = data['features']
                self.metrics = data['metrics']
                self.trained_date = data['trained_date']
                self.model_type = data.get('model_type', 'decision_tree')
                return True
            return False
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            return False
    
    def get_model_info(self):
        """Model hakkında bilgi al."""
        if self.model is None:
            success = self.load()
            if not success:
                return None
        
        # Özellik önemleri
        feature_importance = None
        if self.model_type == "decision_tree":
            try:
                importance_df = self.get_feature_importance()
                if importance_df is not None:
                    feature_importance = importance_df.to_dict('records')
            except:
                pass
        
        # Model bilgisini döndür
        return {
            'metrics': self.metrics,
            'features': self.features,
            'trained_date': self.trained_date.isoformat() if self.trained_date else None,
            'model_type': self.model_type,
            'feature_importance': feature_importance
        } 