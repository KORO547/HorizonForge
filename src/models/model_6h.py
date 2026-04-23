import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

print("=== Модель 6h (предсказание направления через 6 часов) ===\n")

# Топ-признаки из v6
top_features = [
    'bb_lower', 'close_lag_5', 'bb_middle', 'low', 'close_lag_10', 
    'ema_21', 'ema_9', 'close_lag_20', 'close', 'open',
    'bb_upper', 'ema_50', 'close_lag_3', 'ema_9_21_ratio', 
    'price_range', 'vol_10', 'volume_ratio', 'atr_14', 
    'rsi_14', 'hour_sin', 'dayofweek'
]

def train_6h_model(df, symbol):
    X = df[top_features]
    y = df['target']                    # 1 = цена вырастет через 6 часов
    
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
    
    if mean_auc > 0.55:
        print("✅ Хороший сигнал для 6-часового горизонта!")
    elif mean_auc > 0.52:
        print("⚠️  Слабый, но есть сигнал")
    else:
        print("⚠️  Сигнал пока слабый")
    
    return model

# Загружаем данные v6
btc = pd.read_csv('data/features/BTC_USDT_1h_features_v6.csv')
eth = pd.read_csv('data/features/ETH_USDT_1h_features_v6.csv')

btc_model = train_6h_model(btc, "BTC/USDT")
eth_model = train_6h_model(eth, "ETH/USDT")

# Сохраняем модели
os.makedirs('models', exist_ok=True)
btc_model.booster_.save_model('models/lgb_btc_6h.txt')
eth_model.booster_.save_model('models/lgb_eth_6h.txt')

print("\nМодель 6h сохранена!")
print("Теперь можно сделать backtest под 6-часовой горизонт.")