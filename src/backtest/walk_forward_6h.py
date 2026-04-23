import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

print("=== Walk-Forward (Пошаговая проверка во времени) для 6h модели ===\n")

initial_capital = 10000.0
commission = 0.00075          # комиссия 0.075%
min_confidence = 0.57         # минимальная уверенность модели

# Загружаем модели
model_btc = lgb.Booster(model_file='models/lgb_btc_6h.txt')
model_eth = lgb.Booster(model_file='models/lgb_eth_6h.txt')

def run_walk_forward(symbol, model):
    df = pd.read_csv(f'data/features/{symbol}_1h_features_v6.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    top_features = [
        'bb_lower', 'close_lag_5', 'bb_middle', 'low', 'close_lag_10', 
        'ema_21', 'ema_9', 'close_lag_20', 'close', 'open',
        'bb_upper', 'ema_50', 'close_lag_3', 'ema_9_21_ratio', 
        'price_range', 'vol_10', 'volume_ratio', 'atr_14', 
        'rsi_14', 'hour_sin', 'dayofweek'
    ]
    
    X = df[top_features]
    df['prob_up'] = model.predict(X)                    # вероятность роста
    df['predicted_dir'] = (df['prob_up'] >= 0.5).astype(int)
    df['confidence'] = np.where(df['predicted_dir'] == 1, df['prob_up'], 1 - df['prob_up'])
    
    # Walk-Forward параметры
    window_size = 400          # сколько свечей используем для обучения (примерно 16 дней)
    step = 50                  # шаг вперёд (примерно 2 дня)
    
    equity = [initial_capital]
    capital = initial_capital
    positions = []
    
    print(f"Запускаем Walk-Forward для {symbol.replace('_', '/')}...")
    
    for start in range(0, len(df) - window_size, step):
        end = start + window_size
        test_start = end
        test_end = min(end + step, len(df))
        
        if test_end - test_start < 10:
            break
            
        # Тестовый период
        test_df = df.iloc[test_start:test_end].copy()
        
        # Предсказания
        test_X = test_df[top_features]
        test_df['prob_up'] = model.predict(test_X)
        test_df['confidence'] = np.where(test_df['prob_up'] >= 0.5, test_df['prob_up'], 1 - test_df['prob_up'])
        
        # Простая стратегия: держим позицию 6 часов, если уверенность высокая
        position = 0
        for i in range(len(test_df)):
            if test_df['confidence'].iloc[i] >= min_confidence:
                position = 1 if test_df['prob_up'].iloc[i] >= 0.5 else -1
            
            ret = test_df['close'].iloc[i] / test_df['close'].iloc[i-1] - 1 if i > 0 else 0
            strategy_ret = position * ret
            
            if i > 0 and position != 0 and test_df['confidence'].iloc[i] >= min_confidence:
                strategy_ret -= commission   # комиссия при входе
            
            capital *= (1 + strategy_ret)
            equity.append(capital)
    
    df_equity = pd.DataFrame({'timestamp': df['timestamp'].iloc[:len(equity)], 'equity': equity})
    
    total_return = (capital / initial_capital) - 1
    max_drawdown = ((df_equity['equity'] / df_equity['equity'].cummax()) - 1).min()
    
    print(f"\n=== Walk-Forward результаты для {symbol.replace('_', '/')} ===")
    print(f"Итоговая доходность: {total_return*100:.2f}%")
    print(f"Финальный капитал: ${capital:.2f}")
    print(f"Максимальная просадка (Maximum Drawdown): {max_drawdown*100:.2f}%")
    
    # График
    plt.figure(figsize=(12, 6))
    plt.plot(df_equity['timestamp'], df_equity['equity'], label='Стратегия 6h (Walk-Forward)')
    plt.title(f'Walk-Forward Equity Curve - {symbol.replace("_", "/")}')
    plt.xlabel('Дата')
    plt.ylabel('Капитал ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Запускаем
run_walk_forward('BTC_USDT', model_btc)
run_walk_forward('ETH_USDT', model_eth)

print("\nWalk-Forward (Пошаговая проверка во времени) завершён!")