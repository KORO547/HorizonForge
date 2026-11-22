import pandas as pd
import numpy as np
from datetime import datetime
from lightgbm import LGBMClassifier
import joblib
import os

print("=== Обучение полной модели крипты (Цена + Dominance) ===")

# Загружаем данные
file_path = 'data/crypto/crypto_with_macro.csv'
if not os.path.exists(file_path):
    print("❌ Файл с данными не найден!")
    exit()

df = pd.read_csv(file_path)
print(f"Загружено строк: {len(df)}, колонок: {list(df.columns)}")

# Если данных мало — создаём синтетические
if len(df) < 20 or 'BTC_Price' not in df.columns:
    print("⚠️ Создаём синтетические данные для обучения...")
    dates = pd.date_range(end=datetime.now().date(), periods=80)
    df = pd.DataFrame({
        'Date': dates,
        'Brent_Price': np.random.normal(84, 6, 80),
        'Crypto_Mining_Global_TWh': np.random.normal(158, 12, 80),
        'Global_CO2_Mt': np.random.normal(37800, 400, 80),
        'China_Electricity_Price': np.random.normal(0.115, 0.012, 80),
        'Energy_Crisis_Index': np.random.normal(68, 10, 80),
        'BTC_Price': np.random.normal(92000, 8000, 80),
        'BTC_Dominance': np.random.normal(52.5, 3.5, 80)
    })

# Признаки
feature_cols = ['Brent_Price', 'Crypto_Mining_Global_TWh', 'Global_CO2_Mt', 
                'China_Electricity_Price', 'Energy_Crisis_Index']

X = df[feature_cols]

# Целевые переменные
y_price = (df['BTC_Price'].diff() > 0).astype(int)
y_dom = (df['BTC_Dominance'].diff() > 0).astype(int)

# Модели
model_price = LGBMClassifier(n_estimators=700, max_depth=7, learning_rate=0.06, random_state=42)
model_dom = LGBMClassifier(n_estimators=600, max_depth=6, learning_rate=0.07, random_state=42)

model_price.fit(X, y_price)
model_dom.fit(X, y_dom)

# Сохранение
os.makedirs('models', exist_ok=True)
joblib.dump(model_price, 'models/crypto_price_model.pkl')
joblib.dump(model_dom, 'models/crypto_dom_model.pkl')

# Прогноз
latest = X.iloc[-1:].copy()
prob_price = model_price.predict_proba(latest)[0][1] * 100
prob_dom = model_dom.predict_proba(latest)[0][1] * 100

print(f"✅ Модель успешно обучена!")
print(f"Вероятность роста цены BTC     : {prob_price:.1f}%")
print(f"Вероятность роста Dominance   : {prob_dom:.1f}%")