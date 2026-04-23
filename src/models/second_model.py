import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

print("=== Вторая модель: LightGBM (улучшенная версия) ===\n")

# Топ-25 самых сильных признаков
top_features = [
    'dayofweek', 'hour', 'hour_sin', 'close_lag_20', 'bb_lower', 
    'vol_10', 'price_range', 'close_lag_10', 'dc_upper', 'bb_middle',
    'close_lag_5', 'ema_21', 'bb_upper', 'close_lag_3', 'low',
    'return_lag_1', 'atr_14', 'volume_ratio', 'ema_9', 'dc_lower',
    'mom_10', 'vol_20', 'rsi_14', 'return_1', 'close_lag_2'
]

def train_and_evaluate(df, symbol):
    X = df[top_features]
    y = df['target_dir']
    
    print(f"\n=== {symbol} ===")
    print(f"Используем {len(top_features)} признаков")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    accuracies = []
    auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
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
    
    print(f"\nСредняя точность: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
    print(f"Средний AUC: {np.mean(auc_scores):.4f} (±{np.std(auc_scores):.4f})")
    
    return model

# Загружаем данные
btc = pd.read_csv('data/features/BTC_USDT_1h_features.csv')
eth = pd.read_csv('data/features/ETH_USDT_1h_features.csv')

# Обучаем
btc_model = train_and_evaluate(btc, "BTC/USDT")
eth_model = train_and_evaluate(eth, "ETH/USDT")

print("\nВторая модель готова!")
print("Сравниваем с первой версией.")

# Сохраняем модели
os.makedirs('models', exist_ok=True)
btc_model.booster_.save_model('models/lgb_btc.txt')
eth_model.booster_.save_model('models/lgb_eth.txt')