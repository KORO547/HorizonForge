import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

print("=== Модель 6h v2 (с Order Book Imbalance) ===\n")

# Топ-признаки из v6.1
top_features = [
    'bb_lower', 'close_lag_20', 'bb_middle', 'close_lag_5', 'ema_21', 
    'low', 'close_lag_10', 'ema_9', 'ema_50', 'close',
    'bb_upper', 'close_lag_3', 'ema_9_21_ratio', 'price_range', 
    'vol_10', 'volume_ratio', 'atr_14', 'rsi_14', 'hour_sin', 
    'dayofweek', 'ob_imbalance'   # новый признак
]

def train_6h_v2_model(df, symbol):
    X = df[top_features]
    y = df['target']
    
    print(f"\n=== {symbol} ===")
    print(f"Используем {len(top_features)} признаков")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    accuracies = []
    auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=800,
            max_depth=7,
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
        
        print(f"Fold {fold+1}: Accuracy = {acc:.4f} | AUC = {auc:.4f}")
    
    mean_acc = np.mean(accuracies)
    mean_auc = np.mean(auc_scores)
    
    print(f"\nСредняя точность: {mean_acc:.4f} (±{np.std(accuracies):.4f})")
    print(f"Средний AUC: {mean_auc:.4f} (±{np.std(auc_scores):.4f})")
    
    if mean_auc > 0.56:
        print("✅ Хороший сигнал")
    elif mean_auc > 0.53:
        print("⚠️  Слабый, но есть сигнал")
    else:
        print("⚠️  Сигнал пока слабый")
    
    return model

# Загружаем данные v6.1
btc = pd.read_csv('data/features/BTC_USDT_features_v6.1.csv')
eth = pd.read_csv('data/features/ETH_USDT_features_v6.1.csv')

btc_model = train_6h_v2_model(btc, "BTC/USDT")
eth_model = train_6h_v2_model(eth, "ETH/USDT")

# Сохраняем
os.makedirs('models', exist_ok=True)
btc_model.booster_.save_model('models/lgb_btc_6h_v2.txt')
eth_model.booster_.save_model('models/lgb_eth_6h_v2.txt')

print("\nМодель 6h v2 сохранена!")