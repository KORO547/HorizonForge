import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

print("=== Walk-Forward v2 для модели 6h (с Order Book Imbalance) ===\n")

initial_capital = 10000.0
commission = 0.00075
min_confidence = 0.57

# Загружаем модели v2
model_btc = lgb.Booster(model_file='models/lgb_btc_6h_v2.txt')
model_eth = lgb.Booster(model_file='models/lgb_eth_6h_v2.txt')

def run_walk_forward_v2(symbol, model):
    df = pd.read_csv(f'data/features/{symbol}_features_v6.1.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    top_features = [
        'bb_lower', 'close_lag_20', 'bb_middle', 'close_lag_5', 'ema_21', 
        'low', 'close_lag_10', 'ema_9', 'ema_50', 'close',
        'bb_upper', 'close_lag_3', 'ema_9_21_ratio', 'price_range', 
        'vol_10', 'volume_ratio', 'atr_14', 'rsi_14', 'hour_sin', 
        'dayofweek', 'ob_imbalance'
    ]
    
    X = df[top_features]
    df['prob_up'] = model.predict(X)
    df['confidence'] = np.maximum(df['prob_up'], 1 - df['prob_up'])
    
    # Параметры Walk-Forward
    train_size = 400      # размер окна обучения (~16 дней)
    step = 48             # шаг вперёд (~2 дня)
    
    equity = [initial_capital]
    capital = initial_capital
    
    print(f"Запускаем Walk-Forward для {symbol.replace('_', '/')}...")
    
    for start in range(0, len(df) - train_size - 24, step):
        train_end = start + train_size
        test_end = min(train_end + step, len(df))
        
        if test_end - train_end < 12:
            break
            
        test_df = df.iloc[train_end:test_end].copy()
        test_X = test_df[top_features]
        
        test_df['prob_up'] = model.predict(test_X)
        test_df['confidence'] = np.maximum(test_df['prob_up'], 1 - test_df['prob_up'])
        
        position = 0
        for i in range(1, len(test_df)):
            if test_df['confidence'].iloc[i] >= min_confidence:
                position = 1 if test_df['prob_up'].iloc[i] >= 0.5 else -1
            
            ret = test_df['close'].iloc[i] / test_df['close'].iloc[i-1] - 1
            strategy_ret = position * ret
            
            # Комиссия при входе в позицию
            if i > 1 and position != 0 and test_df['confidence'].iloc[i] >= min_confidence and test_df['confidence'].iloc[i-1] < min_confidence:
                strategy_ret -= commission
            
            capital *= (1 + strategy_ret)
            equity.append(capital)
    
    # Метрики
    total_return = (capital / initial_capital) - 1
    equity_series = pd.Series(equity)
    max_drawdown = ((equity_series / equity_series.cummax()) - 1).min()
    
    print(f"\n=== Walk-Forward v2 результаты для {symbol.replace('_', '/')} ===")
    print(f"Итоговая доходность: {total_return*100:.2f}%")
    print(f"Финальный капитал: ${capital:.2f}")
    print(f"Максимальная просадка: {max_drawdown*100:.2f}%")
    
    # График
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'][:len(equity)], equity, label='Стратегия 6h (Walk-Forward v2)')
    plt.title(f'Walk-Forward v2 - {symbol.replace("_", "/")}')
    plt.xlabel('Дата')
    plt.ylabel('Капитал ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Запускаем
run_walk_forward_v2('BTC_USDT', model_btc)
run_walk_forward_v2('ETH_USDT', model_eth)

print("\nWalk-Forward v2 завершён!")