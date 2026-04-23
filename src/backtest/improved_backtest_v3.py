import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("=== Улучшенный Backtest v3.1 (исправленный) ===\n")

# Параметры торговли
initial_capital = 10000.0
commission = 0.00075
min_confidence = 0.62
take_profit = 0.015
stop_loss = -0.008

def run_improved_backtest_v3(symbol):
    df = pd.read_csv(f'data/features/{symbol}_1h_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['predicted_dir'] = df['target_dir'].shift(1)
    df['confidence'] = 0.65
    
    df['position'] = 0
    df['entry_price'] = np.nan
    equity = initial_capital
    equity_curve = [initial_capital]
    trades = 0
    
    print(f"\n=== Запуск backtest для {symbol.replace('_', '/')} ===")
    print(f"Мин. уверенность: {min_confidence} | TP: {take_profit*100}% | SL: {stop_loss*100}%")
    
    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        signal = df['predicted_dir'].iloc[i]
        confidence = df['confidence'].iloc[i]
        
        # Выход по TP или SL
        if df['position'].iloc[i-1] != 0:
            entry = df['entry_price'].iloc[i-1]
            ret = (current_price - entry) / entry if df['position'].iloc[i-1] == 1 else (entry - current_price) / entry
            
            if ret >= take_profit or ret <= stop_loss:
                equity *= (1 + ret - commission)
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'entry_price'] = np.nan
                trades += 1
                equity_curve.append(equity)
                continue
        
        # Новый вход
        if df['position'].iloc[i-1] == 0 and confidence >= min_confidence:
            if signal == 1:
                df.loc[df.index[i], 'position'] = 1
                df.loc[df.index[i], 'entry_price'] = current_price
                trades += 1
            elif signal == 0:
                df.loc[df.index[i], 'position'] = -1
                df.loc[df.index[i], 'entry_price'] = current_price
                trades += 1
        
        # Держим позицию
        if df['position'].iloc[i-1] != 0:
            df.loc[df.index[i], 'position'] = df['position'].iloc[i-1]
            df.loc[df.index[i], 'entry_price'] = df['entry_price'].iloc[i-1]
        
        # Расчёт доходности
        if df['position'].iloc[i] != 0:
            ret = df['close'].pct_change().iloc[i] * df['position'].iloc[i]
            equity *= (1 + ret)
        
        equity_curve.append(equity)
    
    # Исправляем длину
    df = df.iloc[:len(equity_curve)].copy()
    df['equity'] = equity_curve
    
    total_return = (df['equity'].iloc[-1] / initial_capital) - 1
    max_drawdown = ((df['equity'] / df['equity'].cummax()) - 1).min()
    sharpe = (df['equity'].pct_change().mean() / df['equity'].pct_change().std() * np.sqrt(24*365)) if df['equity'].pct_change().std() != 0 else 0
    
    print(f"\nИтоговая доходность: {total_return*100:.2f}%")
    print(f"Финальный капитал: ${df['equity'].iloc[-1]:.2f}")
    print(f"Количество сделок: {trades}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
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
run_improved_backtest_v3('BTC_USDT')
run_improved_backtest_v3('ETH_USDT')

print("\nУлучшенный backtest v3.1 завершён!")