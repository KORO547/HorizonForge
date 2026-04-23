import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
import os

print("=== Глобальная модель зависимости v2 (186 акций) ===\n")

df = pd.read_csv('data/features/stocks/global_panel_with_deps.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Загружено: {df.shape} строк | Акций: {df['ticker'].nunique()}")

feature_cols = [
    'rsi_14', 'macd', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower',
    'atr_14', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'price_range', 'vol_10', 'volume_ratio',
    'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_10', 'close_lag_20',
    'market_return', 'sector_return',
    'relative_to_market', 'relative_to_sector',
    'market_return_lag1', 'sector_return_lag1',
    'dayofweek'
]

X = df[feature_cols]
y = df['target']

print(f"Используем {len(feature_cols)} признаков\n")

tscv = TimeSeriesSplit(n_splits=5)
auc_scores = []
acc_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=900,
        max_depth=8,
        learning_rate=0.028,
        subsample=0.78,
        colsample_bytree=0.78,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_val)[:, 1]
    pred_class = (pred_proba > 0.5).astype(int)
    
    auc = roc_auc_score(y_val, pred_proba)
    acc = accuracy_score(y_val, pred_class)
    
    auc_scores.append(auc)
    acc_scores.append(acc)
    
    print(f"Fold {fold}: AUC = {auc:.4f} | Accuracy = {acc:.4f}")

print("\n" + "="*70)
print("РЕЗУЛЬТАТЫ ГЛОБАЛЬНОЙ МОДЕЛИ ЗАВИСИМОСТЕЙ (186 акций)")
print("="*70)
print(f"Средний AUC     : {np.mean(auc_scores):.4f} (±{np.std(auc_scores):.4f})")
print(f"Средняя Accuracy: {np.mean(acc_scores):.4f} (±{np.std(acc_scores):.4f})")
print(f"Лучший AUC      : {max(auc_scores):.4f}")

# Финальная модель
final_model = lgb.LGBMClassifier(
    n_estimators=1200,
    max_depth=9,
    learning_rate=0.025,
    subsample=0.80,
    colsample_bytree=0.80,
    random_state=42,
    verbose=-1
)
final_model.fit(X, y)

import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(final_model, 'models/global_dependency_model_v2.pkl')

print("\nГлобальная модель v2 успешно обучена и сохранена!")
print("Файл: models/global_dependency_model_v2.pkl")