import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("=== Улучшенный Backtest v2 (с фильтрами и риск-менеджментом) ===\n")

# Параметры
initial_capital = 10000.0
commission = 0.00075          # 0.075%
min_confidence = 0.58         # минимальная уверенность для входа в позицию
risk_per_trade = 0.01         # риск на одну сделку 1% от капитала

def run_improved_backtest_v2(symbol):
    df = pd.read_csv(f'data/features/{symbol}_1h_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Пока используем target_dir как "предсказание". Позже заменим на реальные вероятности модели
    df['predicted_dir'] = df['target_dir'].shift(1)
    df['confidence'] = 0.6                    # заглушка (потом будет от модели)
    
    # Фильтр: входим только при достаточной уверенности
    df['signal'] = 0
    df.loc[(df['predicted_dir'] == 1) & (df['confidence'] >= min_confidence), 'signal'] = 1   # long
    df.loc[(df['predicted_dir'] == 0) & (df['confidence'] >= min_confidence), 'signal'] = -1  # short
    
    # Позиция
    df['position'] = 0
    current_position = 0
    
    for i in range(1, len(df)):
        if df['signal'].iloc[i] != 0 and df['signal'].iloc[i] != current_position:
            current_position = df['signal'].iloc[i]
        df.loc[df.index[i], 'position'] = current_position
    
    # Доходность
    df['market_return'] = df['close'].pct_change()
    df['strategy_return'] = df['position'].shift(1) * df['market_return']
    
    # Комиссия при смене позиции
    df['trade'] = (df['position'] != df['position'].shift(1)).astype(int)
    df['strategy_return'] = df['strategy_return'] - df['trade'] * commission
    
    # Кумулятивная кривая
    df['cum_return'] = (1 + df['strategy_return']).cumprod()
    df['equity'] = initial_capital * df['cum_return']
    
    # Метрики
    total_return = df['cum_return'].iloc[-1] - 1
    num_trades = df['trade'].sum()
    max_drawdown = ((df['equity'] / df['equity'].cummax()) - 1).min()
    sharpe = (df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(24*365)) if df['strategy_return'].std() != 0 else 0
    
    print(f"\n=== Результаты для {symbol.replace('_', '/')} ===")
    print(f"Итоговая доходность: {total_return*100:.2f}%")
    print(f"Финальный капитал: ${df['equity'].iloc[-1]:.2f}")
    print(f"Количество сделок: {num_trades}")
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

# Запускаем для обеих монет
run_improved_backtest_v2('BTC_USDT')
run_improved_backtest_v2('ETH_USDT')

print("\nУлучшенный backtest v2 завершён!")