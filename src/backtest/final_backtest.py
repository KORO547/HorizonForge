import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

print("=== Финальный Backtest с реальной моделью ===\n")

# Параметры
initial_capital = 10000.0
commission = 0.00075        # 0.075%
risk_per_trade = 0.01       # риск 1% на сделку
min_confidence = 0.57       # минимальная уверенность модели

# Загружаем модель
model_btc = lgb.Booster(model_file='models/lgb_btc_final.txt')
model_eth = lgb.Booster(model_file='models/lgb_eth_final.txt')

def run_final_backtest(symbol, model):
    df = pd.read_csv(f'data/features/{symbol}_1h_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    top_features = [
        'dayofweek', 'hour', 'hour_sin', 'close_lag_20', 'bb_lower', 
        'vol_10', 'price_range', 'close_lag_10', 'bb_middle',
        'close_lag_5', 'ema_21', 'bb_upper', 'close_lag_3', 'low',
        'return_lag_1', 'atr_14', 'volume_ratio', 'ema_9', 
        'vol_20', 'rsi_14', 'return_1', 'close_lag_2',
        'ema_21_50_ratio', 'close_lag_1', 'open'
    ]
    
    X = df[top_features]
    
    # Получаем предсказания и вероятности от модели
    df['prob_up'] = model.predict(X)
    df['predicted_dir'] = (df['prob_up'] >= 0.5).astype(int)
    df['confidence'] = df['prob_up'].where(df['predicted_dir'] == 1, 1 - df['prob_up'])
    
    # Позиция
    df['position'] = 0
    current_position = 0
    capital = initial_capital
    equity = [capital]
    
    for i in range(1, len(df)):
        signal = 0
        if df['confidence'].iloc[i] >= min_confidence:
            signal = 1 if df['predicted_dir'].iloc[i] == 1 else -1
        
        # Меняем позицию только если сигнал сильный
        if signal != current_position:
            current_position = signal
        
        df.loc[df.index[i], 'position'] = current_position
        
        # Доходность
        ret = df['close'].iloc[i] / df['close'].iloc[i-1] - 1
        strategy_ret = current_position * ret
        
        # Комиссия
        if df['position'].iloc[i] != df['position'].iloc[i-1]:
            strategy_ret -= commission
        
        capital *= (1 + strategy_ret)
        equity.append(capital)
    
    df['equity'] = equity
    
    # Метрики
    total_return = (capital / initial_capital) - 1
    max_drawdown = ((df['equity'] / df['equity'].cummax()) - 1).min()
    num_trades = (df['position'].diff() != 0).sum()
    
    print(f"\n=== Результаты для {symbol.replace('_', '/')} ===")
    print(f"Итоговая доходность: {total_return*100:.2f}%")
    print(f"Финальный капитал: ${capital:.2f}")
    print(f"Количество сделок: {num_trades}")
    print(f"Максимальная просадка: {max_drawdown*100:.2f}%")
    
    # График
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['equity'], label='Стратегия')
    plt.title(f'Equity Curve - {symbol.replace("_", "/")}')
    plt.xlabel('Дата')
    plt.ylabel('Капитал ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Запускаем
run_final_backtest('BTC_USDT', model_btc)
run_final_backtest('ETH_USDT', model_eth)

print("\nФинальный backtest завершён!")