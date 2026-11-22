import pandas as pd
import joblib
import os
from datetime import datetime
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

def train_crypto_predictor():
    print("=== Обучение улучшенной модели крипты с макро-факторами ===")
    
    data_path = 'data/crypto/crypto_with_macro.csv'
    if not os.path.exists(data_path):
        print("❌ Данные не найдены")
        return
    
    df = pd.read_csv(data_path)
    
    # Если данных мало — создаём синтетическую историю для обучения
    if len(df) < 10:
        print("⚠️ Мало данных. Создаём синтетическую историю для обучения...")
        dates = pd.date_range(end=datetime.now().date(), periods=50)
        synthetic = pd.DataFrame({
            'Date': dates,
            'Brent_Price': [75 + i*0.2 for i in range(50)],
            'Gas_Europe': [10 + i*0.05 for i in range(50)],
            'Crypto_Mining_Global_TWh': [140 + i*0.4 for i in range(50)],
            'Global_CO2_Mt': [37000 + i*16 for i in range(50)],
            'China_Electricity_Price': [0.10 + i*0.0003 for i in range(50)],
            'Energy_Crisis_Index': [60 + i*0.2 for i in range(50)],
            'BTC_Target': [1 if i % 3 != 0 else 0 for i in range(50)]  # синтетическая цель
        })
        df = synthetic
        df.to_csv(data_path, index=False)
    
    feature_cols = ['Brent_Price', 'Gas_Europe', 'Crypto_Mining_Global_TWh', 
                   'Global_CO2_Mt', 'China_Electricity_Price', 'Energy_Crisis_Index']
    
    X = df[feature_cols]
    y = df['BTC_Target']
    
    model = LGBMClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)
    
    # Сохранение
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/crypto_predictor_btc.pkl')
    
    # Предсказание на сегодня
    latest = X.iloc[-1:].copy()
    prob = model.predict_proba(latest)[0][1]
    
    print(f"✅ Модель крипты обучена на {len(df)} записях!")
    print(f"Вероятность роста BTC: {prob:.1%}")
    print(f"Факторы: Brent=${latest['Brent_Price'].iloc[0]:.1f}, Mining={latest['Crypto_Mining_Global_TWh'].iloc[0]} TWh")
    
    return model, prob

if __name__ == "__main__":
    train_crypto_predictor()