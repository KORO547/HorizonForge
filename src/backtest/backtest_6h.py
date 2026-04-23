import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

print("=== Backtest для 6-часовой модели ===\n")

initial_capital = 10000.0
commission = 0.00075          # 0.075%
min_confidence = 0.56         # порог уверенности

# Загружаем модели
model_btc = lgb.Booster(model_file='models/lgb_btc_6h.txt')
model_eth = lgb.Booster(model_file='models/lgb_eth_6h.txt')

def run_6h_backtest(symbol, model):
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
    
    df['prob_up'] = model.predict(X)
    df['predicted_dir'] = (df['prob_up'] >= 0.5).astype(int)
    df['confidence'] = np.where(df['predicted_dir'] == 1, df['prob_up'], 1 - df['prob_up'])
    
    # Позиция держится 6 часов
    df['position'] = 0
    capital = initial_capital
    equity = [capital]
    position_until = 0
    
    for i in range(len(df)):
        if i >= position_until and df['confidence'].iloc[i] >= min_confidence:
            # Открываем новую позицию на 6 часов
            df.loc[df.index[i], 'position'] = 1 if df['predicted_dir'].iloc[i] == 1 else -1
            position_until = i + 6   # держим 6 свечей
        
        # Если позиция активна
        if df['position'].iloc[i] == 0 and i < position_until:
            df.loc[df.index[i], 'position'] = df['position'].iloc[i-1]
        
        # Расчёт доходности
        if i > 0:
            ret = df['close'].iloc[i] / df['close'].iloc[i-1] - 1
            strategy_ret = df['position'].iloc[i-1] * ret
            
            # Комиссия только при открытии
            if df['position'].iloc[i] != df['position'].iloc[i-1]:
                strategy_ret -= commission
                
            capital *= (1 + strategy_ret)
        
        equity.append(capital)
    
    df['equity'] = equity[:len(df)]
    
    total_return = (capital / initial_capital) - 1
    max_drawdown = ((df['equity'] / df['equity'].cummax()) - 1).min()
    num_trades = (df['position'].diff() != 0).sum()
    
    print(f"\n=== Результаты 6h для {symbol.replace('_', '/')} ===")
    print(f"Итоговая доходность: {total_return*100:.2f}%")
    print(f"Финальный капитал: ${capital:.2f}")
    print(f"Количество сделок: {num_trades}")
    print(f"Максимальная просадка: {max_drawdown*100:.2f}%")
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['equity'], label='Стратегия 6h')
    plt.title(f'Equity Curve 6h - {symbol.replace("_", "/")}')
    plt.xlabel('Дата')
    plt.ylabel('Капитал ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Запускаем
run_6h_backtest('BTC_USDT', model_btc)
run_6h_backtest('ETH_USDT', model_eth)

print("\nBacktest 6h завершён!")