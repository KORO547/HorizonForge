import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

print("=== Третья модель: LightGBM + Purged Walk-Forward ===\n")

# Топ-25 самых сильных признаков (на основе предыдущего анализа)
top_features = [
    'dayofweek', 'hour', 'hour_sin', 'close_lag_20', 'bb_lower',
    'vol_10', 'price_range', 'close_lag_10', 'dc_upper', 'bb_middle',
    'close_lag_5', 'ema_21', 'bb_upper', 'close_lag_3', 'low',
    'return_lag_1', 'atr_14', 'volume_ratio', 'ema_9', 'dc_lower',
    'mom_10', 'vol_20', 'rsi_14', 'return_1', 'close_lag_2'
]

def purged_walk_forward(df, symbol, gap=12):
    """Purged Walk-Forward с зазором, чтобы избежать утечки данных"""
    X = df[top_features]
    y = df['target_dir']
    
    print(f"\n=== {symbol} ===")
    print(f"Используем {len(top_features)} признаков | Gap = {gap} часов")
    
    n_samples = len(X)
    accuracies = []
    auc_scores = []
    
    # Ручной Purged Walk-Forward
    train_size = int(n_samples * 0.6)   # 60% на обучение
    step = int(n_samples * 0.08)        # шаг ~8%
    
    fold = 1
    for start in range(train_size, n_samples - step, step):
        train_end = start - gap         # зазор, чтобы не утечь информацию
        if train_end <= 0:
            continue
            
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[start:start+step], y.iloc[start:start+step]
        
        if len(X_val) < 50:
            break
            
        model = lgb.LGBMClassifier(
            n_estimators=500,
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
        
        print(f"Fold {fold}: Accuracy = {acc:.4f} | AUC = {auc:.4f} | Val samples: {len(X_val)}")
        fold += 1
    
    if accuracies:
        print(f"\nСредняя точность: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
        print(f"Средний AUC: {np.mean(auc_scores):.4f} (±{np.std(auc_scores):.4f})")
    else:
        print("Не удалось сформировать валидационные фолды")
    
    return model

# Загружаем данные
btc = pd.read_csv('data/features/BTC_USDT_1h_features.csv')
eth = pd.read_csv('data/features/ETH_USDT_1h_features.csv')

# Обучаем
btc_model = purged_walk_forward(btc, "BTC/USDT")
eth_model = purged_walk_forward(eth, "ETH/USDT")

print("\nТретья модель завершена.")
print("Теперь мы видим более честную оценку.")

# Сохраняем модели
os.makedirs('models', exist_ok=True)
btc_model.booster_.save_model('models/lgb_btc_final.txt')
eth_model.booster_.save_model('models/lgb_eth_final.txt')