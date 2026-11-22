import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from lightgbm import LGBMClassifier
import joblib
import os

print("=== Politico-Economic Memory Model v2 (с реальной историей + ФРС) ===")

# === 1. Скачиваем реальные исторические данные ===
print("Скачиваем исторические данные BTC и макро...")
btc = yf.download("BTC-USD", period="2y", interval="1d")['Close'].reset_index()
btc.columns = ['Date', 'BTC_Price']

fed = yf.download("^IRX", period="2y", interval="1d")['Close'].reset_index()  # приближение к ставке ФРС
fed.columns = ['Date', 'Fed_Rate_Proxy']

# Объединяем
df = pd.merge(btc, fed, on='Date', how='left')
df['Fed_Rate_Proxy'] = df['Fed_Rate_Proxy'].fillna(method='ffill')

print(f"Загружено {len(df)} дней исторических данных")

# === 2. Добавляем синтетические макро-факторы (пока) ===
np.random.seed(42)
n = len(df)
df['Brent_Price'] = np.random.normal(82, 8, n)
df['Crypto_Mining_Global_TWh'] = np.random.normal(155, 18, n)
df['Global_CO2_Mt'] = np.random.normal(37700, 600, n)
df['China_Electricity_Price'] = np.random.normal(0.113, 0.017, n)
df['Energy_Crisis_Index'] = np.random.normal(67, 14, n)
df['US_China_Tension'] = np.random.normal(0.5, 0.23, n).clip(0.1, 0.95)
df['Election_Proximity'] = np.exp(-np.abs(np.arange(n) - n//2) / 90)

# === 3. Глубокая историческая память ===
lags = [1, 3, 7, 14, 30, 60, 90]
for lag in lags:
    df[f'BTC_Price_lag_{lag}'] = df['BTC_Price'].shift(lag)
    df[f'Brent_lag_{lag}'] = df['Brent_Price'].shift(lag)

df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
df['month'] = pd.to_datetime(df['Date']).dt.month
df['year_cycle'] = np.sin(2 * np.pi * pd.to_datetime(df['Date']).dt.dayofyear / 365)

# === 4. Признаки ===
feature_cols = ['Brent_Price', 'Crypto_Mining_Global_TWh', 'Global_CO2_Mt',
                'China_Electricity_Price', 'Energy_Crisis_Index', 'Fed_Rate_Proxy',
                'US_China_Tension', 'Election_Proximity',
                'day_of_week', 'month', 'year_cycle']

for lag in lags:
    feature_cols.extend([f'BTC_Price_lag_{lag}', f'Brent_lag_{lag}'])

X = df[feature_cols].fillna(method='bfill')
y_price = (df['BTC_Price'].diff() > 0).astype(int)

# === 5. Обучение ===
model = LGBMClassifier(
    n_estimators=1100,
    max_depth=9,
    learning_rate=0.04,
    num_leaves=96,
    min_child_samples=30,
    reg_alpha=0.6,
    random_state=42
)

model.fit(X, y_price)

# Сохранение
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/crypto_geopolitical_model_v2.pkl')

# Прогноз
latest = X.iloc[-1:].copy()
prob_price = model.predict_proba(latest)[0][1] * 100

print(f"\n✅ Модель v2 успешно обучена на реальной истории!")
print(f"Вероятность роста цены BTC : {prob_price:.1f}%")
print(f"Использовано дней истории  : {len(df)}")