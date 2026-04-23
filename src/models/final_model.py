import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

print("=== Финальная модель LightGBM (исправленная версия) ===\n")

# Только те признаки, которые точно есть в v5.0
top_features = [
    'dayofweek', 'hour', 'hour_sin', 'close_lag_20', 'bb_lower', 
    'vol_10', 'price_range', 'close_lag_10', 'bb_middle',
    'close_lag_5', 'ema_21', 'bb_upper', 'close_lag_3', 'low',
    'return_lag_1', 'atr_14', 'volume_ratio', 'ema_9', 
    'vol_20', 'rsi_14', 'return_1', 'close_lag_2',
    'ema_21_50_ratio', 'close_lag_1', 'open'
]

def train_final_model(df, symbol):
    X = df[top_features]
    y = df['target_dir']
    
    print(f"\n=== {symbol} ===")
    print(f"Используем {len(top_features)} признаков")
    
    # Правильный TimeSeriesSplit из sklearn
    tscv = TimeSeriesSplit(n_splits=5)
    
    accuracies = []
    auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.025,
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
        
        print(f"Fold {fold+1}: Accuracy = {acc:.4f} | AUC = {auc:.4f}")
    
    mean_acc = np.mean(accuracies)
    mean_auc = np.mean(auc_scores)
    
    print(f"\nСредняя точность: {mean_acc:.4f} (±{np.std(accuracies):.4f})")
    print(f"Средний AUC: {mean_auc:.4f} (±{np.std(auc_scores):.4f})")
    
    if mean_auc > 0.53:
        print("✅ Есть слабый сигнал!")
    else:
        print("⚠️  Сигнал пока очень слабый.")
    
    return model

# Загружаем данные
btc = pd.read_csv('data/features/BTC_USDT_1h_features.csv')
eth = pd.read_csv('data/features/ETH_USDT_1h_features.csv')

btc_model = train_final_model(btc, "BTC/USDT")
eth_model = train_final_model(eth, "ETH/USDT")

print("\nФинальная модель завершена.")

# Сохраняем модели
os.makedirs('models', exist_ok=True)
btc_model.booster_.save_model('models/lgb_btc_final.txt')
eth_model.booster_.save_model('models/lgb_eth_final.txt')