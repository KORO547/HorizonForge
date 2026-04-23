import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

print("=== Модели для акций по секторам ===\n")

top_features = [
    'rsi_14', 'macd', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower',
    'atr_14', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'price_range', 'vol_10', 'volume_ratio', 'dayofweek',
    'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_10', 'close_lag_20'
]

def train_sector_model(df, ticker):
    X = df[top_features]
    y = df['target']
    
    print(f"\n=== {ticker} ===")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    accuracies = []
    auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.75,
            colsample_bytree=0.75,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        pred_proba = model.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, pred)
        auc = roc_auc_score(y_val, pred_proba)
        
        accuracies.append(acc)
        auc_scores.append(auc)
        
        print(f"Fold {fold+1}: Acc = {acc:.4f} | AUC = {auc:.4f}")
    
    mean_auc = np.mean(auc_scores)
    print(f"Средний AUC: {mean_auc:.4f}")
    
    return mean_auc

# Список всех файлов
stock_dir = 'data/features/stocks'
results = []

for filename in os.listdir(stock_dir):
    if filename.endswith('_features.csv'):
        ticker = filename.replace('_features.csv', '')
        df = pd.read_csv(f"{stock_dir}/{filename}")
        
        auc = train_sector_model(df, ticker)
        results.append((ticker, auc))

# Сортируем по качеству сигнала
results.sort(key=lambda x: x[1], reverse=True)

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ ПО СЕКТОРАМ (сортировка по AUC):")
print("="*60)
for ticker, auc in results:
    print(f"{ticker:6} → AUC = {auc:.4f} {'← Лучший' if auc > 0.57 else ''}")

print("\nГотово! Самые перспективные акции выделены.")