import os
import sys
import argparse
from dotenv import load_dotenv

# Ana dizini yola ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Çevre değişkenlerini yükle
load_dotenv()

from app.db.database import SessionLocal
from app.services.ml_service import SalesForecastModel

def train_model(model_type="decision_tree", verbose=True):
    """Satış tahmin modelini eğit."""
    # Veritabanı oturumu oluştur
    db = SessionLocal()
    
    try:
        # Model örneği oluştur
        model = SalesForecastModel()
        
        # Modeli belirtilen parametrelerle eğit
        if verbose:
            print(f"{model_type} modeli eğitiliyor...")
            print("Bu işlem biraz zaman alabilir...")
            
        metrics = model.train(
            db,
            model_type=model_type
        )
        
        # Hata kontrolü
        if "error" in metrics:
            print(f"❌ Model eğitim hatası: {metrics['error']}")
            return
        
        # Metrikleri yazdır
        if verbose:
            print("\n✅ Model başarıyla eğitildi!")
            print("\nModel Metrikleri:")
            print(f"R² Score: {metrics['r2_score']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"Doğruluk: {metrics['accuracy']:.4f} (eşik: {metrics['threshold']:.2f})")
            print(f"Örnek sayısı: {metrics['n_samples']}")
            
            # Model bilgisini al
            model_info = model.get_model_info()
            
            print(f"\nModel tipi: {model_info['model_type']}")
            
            # Özellik önemlerini yazdır (eğer varsa)
            if model_info.get('feature_importance'):
                print("\nEn önemli 10 özellik:")
                for i, feat in enumerate(model_info['feature_importance'][:10], 1):
                    print(f"{i}. {feat['feature']}: {feat['importance']:.4f}")
                    
            print(f"\nEğitim tarihi: {model_info['trained_date']}")
        
        return metrics
        
    finally:
        db.close()

if __name__ == "__main__":
    # Argüman ayrıştırıcı oluştur
    parser = argparse.ArgumentParser(description='Satış tahmin modeli eğitimi')
    parser.add_argument('--model', type=str, 
                      choices=['decision_tree', 'linear', 'knn', 'logistic'],
                      default='decision_tree', help='Eğitilecek model tipi')
    
    # Argümanları ayrıştır
    args = parser.parse_args()
    
    # Modeli belirtilen argümanlarla eğit
    train_model(model_type=args.model) 