import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

print("=== Модель с макро факторами (6h) ===\n")

# Признаки + макро
top_features = [
    'bb_lower', 'close_lag_20', 'bb_middle', 'close_lag_5', 'ema_21', 
    'low', 'close_lag_10', 'ema_9', 'ema_50', 'close',
    'bb_upper', 'close_lag_3', 'price_range', 'vol_10', 'atr_14', 
    'rsi_14', 'hour_sin', 'dayofweek', 'ob_imbalance', 'ob_imbalance_5',
    'Fed Interest Rate', 'CPI Inflation', 'Unemployment Rate', 
    '10-Year Treasury Yield', 'VIX Volatility Index',
    'fed_rate_change', 'vix_change', 'yield_curve'
]

def train_macro_model(df, symbol):
    available_features = [col for col in top_features if col in df.columns]
    X = df[available_features]
    y = df['target']
    
    print(f"\n=== {symbol} ===")
    print(f"Используем {len(available_features)} признаков (включая макро)")
    
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
        print("✅ Хороший сигнал с макро")
    else:
        print("⚠️  Сигнал слабый")
    
    return model

# Обучаем для трёх пар
pairs = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']

for pair in pairs:
    df = pd.read_csv(f'data/features/{pair}_features_v6.3_macro.csv')
    model = train_macro_model(df, pair.replace('_', '/'))
    
    os.makedirs('models', exist_ok=True)
    model.booster_.save_model(f'models/lgb_{pair}_macro.txt')
    print(f"Модель с макро для {pair} сохранена.\n")

print("Обучение моделей с макро факторами завершено!")