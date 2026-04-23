import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

print("=== Первая модель: XGBoost ===\n")

# Загружаем данные
btc = pd.read_csv('data/features/BTC_USDT_1h_features.csv')
eth = pd.read_csv('data/features/ETH_USDT_1h_features.csv')

# Берём только топ-признаки + target
feature_cols = ['dayofweek', 'hour', 'hour_sin', 'close_lag_20', 'bb_lower', 
                'vol_10', 'price_range', 'close_lag_10', 'dc_upper', 'bb_middle',
                'close_lag_5', 'ema_21', 'bb_upper', 'close_lag_3', 'low',
                'return_lag_1', 'atr_14', 'volume_ratio']

# Функция для обучения и оценки
def train_and_evaluate(df, symbol):
    X = df[feature_cols]
    y = df['target_dir']                    # 1 = рост, 0 = падение
    
    # TimeSeriesSplit — правильный способ для временных рядов
    tscv = TimeSeriesSplit(n_splits=5)
    
    accuracies = []
    
    print(f"\n=== Результаты для {symbol} ===")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        
        acc = accuracy_score(y_val, pred)
        accuracies.append(acc)
        
        print(f"Fold {fold+1}: Accuracy = {acc:.4f}")
    
    print(f"Средняя точность: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
    return model

# Обучаем модели
btc_model = train_and_evaluate(btc, "BTC/USDT")
eth_model = train_and_evaluate(eth, "ETH/USDT")

print("\nПервая модель готова!")
print("Теперь мы можем оценить, есть ли у нас хоть какое-то преимущество над случайным угадыванием (50%).")

# Сохраняем модели (пока просто для будущего)
os.makedirs('models', exist_ok=True)
btc_model.save_model('models/xgb_btc.json')
eth_model.save_model('models/xgb_eth.json')