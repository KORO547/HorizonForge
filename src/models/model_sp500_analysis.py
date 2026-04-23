import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import os
from tqdm import tqdm

print("=== Финальный анализ по 158 акциям ===\n")

top_features = [
    'rsi_14', 'macd', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower',
    'atr_14', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'price_range', 'vol_10', 'volume_ratio', 'dayofweek',
    'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_10', 'close_lag_20'
]

input_dir = 'data/features/stocks/sp500'
results = []

files = [f for f in os.listdir(input_dir) if f.endswith('_features.csv')]

print(f"Найдено {len(files)} акций. Обучаем модели...\n")

for filename in tqdm(files):
    ticker = filename.replace('_features.csv', '')
    
    try:
        df = pd.read_csv(f"{input_dir}/{filename}")
        X = df[top_features]
        y = df['target']
        
        if len(X) < 100:
            continue
            
        tscv = TimeSeriesSplit(n_splits=5)
        auc_scores = []
        
        for train_idx, val_idx in tscv.split(X):
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
            pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, pred_proba)
            auc_scores.append(auc)
        
        mean_auc = np.mean(auc_scores)
        results.append((ticker, mean_auc, len(X)))
        
    except Exception as e:
        print(f"Ошибка при обработке {ticker}: {e}")

results.sort(key=lambda x: x[1], reverse=True)

print("\n" + "="*90)
print("ТОП-20 АКЦИЙ ПО СИГНАЛУ (из 158 тикеров)")
print("="*90)

for rank, (ticker, auc, days) in enumerate(results[:20], 1):
    print(f"{rank:2d}. {ticker:6} → AUC = {auc:.4f} | дней = {days}")

print("\n" + "="*90)
print(f"Всего проанализировано: {len(results)} акций")
print(f"Лучший результат: {results[0][0]} с AUC = {results[0][1]:.4f}")
print(f"Средний AUC по всем акциям: {np.mean([r[1] for r in results]):.4f}")

df_results = pd.DataFrame(results, columns=['ticker', 'auc', 'days'])
df_results.to_csv('results/sp500_model_performance_158.csv', index=False)
print("\nРезультаты сохранены в results/sp500_model_performance_158.csv")