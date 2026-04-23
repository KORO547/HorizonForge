import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import joblib
import os

print("=== Улучшенная глобальная модель с климатическими взаимодействиями ===\n")

df = pd.read_csv('data/features/stocks/global_panel_with_climate.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Данные загружены: {df.shape} строк, {df['ticker'].nunique()} акций")

# === Создаём умные климатические взаимодействия ===
df['temp_drought'] = df['global_temp_anomaly'] * df['drought_index']
df['precip_drought'] = df['precipitation_anomaly'] * df['drought_index']
df['extreme_weather_score'] = df['extreme_heat_days'] + df['extreme_rain_days']

# Лаггированные климатические эффекты (влияние погоды с задержкой 1-5 дней)
for lag in [1, 3, 5]:
    df[f'temp_anomaly_lag{lag}'] = df.groupby('ticker')['global_temp_anomaly'].shift(lag)
    df[f'drought_lag{lag}'] = df.groupby('ticker')['drought_index'].shift(lag)
    df[f'precip_lag{lag}'] = df.groupby('ticker')['precipitation_anomaly'].shift(lag)

# Секторные взаимодействия с климатом (пример)
df['is_agri'] = df['ticker'].isin(['CTVA','MOS','CF','NTR','ADM','BG','FMC','AGCO']).astype(int)
df['is_energy'] = df['ticker'].isin(['XOM','CVX','COP','SLB','EOG']).astype(int)
df['is_green'] = df['ticker'].isin(['ENPH','FSLR','BEPC','PLUG','BLDP','SEDG','RUN']).astype(int)

df['climate_agri_impact'] = df['is_agri'] * (df['drought_index'] + df['global_temp_anomaly'])
df['climate_energy_impact'] = df['is_energy'] * df['global_temp_anomaly']
df['climate_green_impact'] = df['is_green'] * df['global_temp_anomaly']

# Финальный список признаков
feature_cols = [
    # Базовые технические и рыночные
    'rsi_14', 'macd', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower',
    'atr_14', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'price_range', 'vol_10', 'volume_ratio',
    'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_10', 'close_lag_20',
    'market_return', 'sector_return',
    'relative_to_market', 'relative_to_sector',
    'market_return_lag1', 'sector_return_lag1',
    'dayofweek',
    
    # Климатические
    'global_temp_anomaly', 'us_temp_anomaly', 'precipitation_anomaly', 'drought_index',
    'extreme_heat_days', 'extreme_rain_days',
    'temp_drought', 'precip_drought', 'extreme_weather_score',
    
    # Лаггированные климатические
    'temp_anomaly_lag1', 'temp_anomaly_lag3', 'temp_anomaly_lag5',
    'drought_lag1', 'drought_lag3', 'drought_lag5',
    
    # Взаимодействия климат × сектор
    'climate_agri_impact', 'climate_energy_impact', 'climate_green_impact'
]

X = df[feature_cols]
y = df['target']

print(f"Используем {len(feature_cols)} признаков (с климатическими взаимодействиями)\n")

# Обучение с TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=9,
        learning_rate=0.028,
        subsample=0.78,
        colsample_bytree=0.78,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred_proba)
    auc_scores.append(auc)
    
    print(f"Fold {fold}: AUC = {auc:.4f}")

print("\n" + "="*70)
print("РЕЗУЛЬТАТЫ УЛУЧШЕННОЙ МОДЕЛИ С КЛИМАТОМ")
print("="*70)
print(f"Средний AUC : {np.mean(auc_scores):.4f} (±{np.std(auc_scores):.4f})")

# Сохраняем финальную модель
final_model = lgb.LGBMClassifier(
    n_estimators=1200,
    max_depth=10,
    learning_rate=0.025,
    subsample=0.80,
    colsample_bytree=0.80,
    random_state=42,
    verbose=-1
)
final_model.fit(X, y)

os.makedirs('models', exist_ok=True)
joblib.dump(final_model, 'models/global_model_climate_enhanced.pkl')

print("\nУлучшенная модель с климатическими взаимодействиями сохранена!")
print("Файл: models/global_model_climate_enhanced.pkl")