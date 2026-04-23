import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("=== Улучшенный Backtest с фильтрами ===\n")

# Параметры торговли
initial_capital = 10000.0      # начальный капитал ($)
commission = 0.00075           # комиссия 0.075% (Binance spot)
min_confidence = 0.55          # минимальная уверенность модели (пока используем вероятность)

def run_improved_backtest(symbol):
    df = pd.read_csv(f'data/features/{symbol}_1h_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Загружаем предсказания модели (пока используем target_dir как "идеальное" предсказание)
    # В будущем здесь будут реальные вероятности от модели
    df['predicted_dir'] = df['target_dir'].shift(1)          # сдвиг, чтобы не смотреть в будущее
    df['confidence'] = 0.6                                   # пока заглушка (потом заменим на вероятность модели)
    
    # Фильтр: торгуем только если уверенность выше порога
    df['position'] = 0
    df.loc[(df['predicted_dir'] == 1) & (df['confidence'] >= min_confidence), 'position'] = 1   # long
    df.loc[(df['predicted_dir'] == 0) & (df['confidence'] >= min_confidence), 'position'] = -1  # short (или 0)
    
    # Доходность стратегии
    df['market_return'] = df['close'].pct_change()
    df['strategy_return'] = df['position'].shift(1) * df['market_return']
    
    # Комиссия при смене позиции
    df['trade'] = df['position'].diff().abs()
    df['strategy_return'] = df['strategy_return'] - (df['trade'] * commission)
    
    # Кумулятивная кривая капитала
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
    print(f"Количество сделок: {num_trades:.0f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Максимальная просадка: {max_drawdown*100:.2f}%")
    
    # Сохраняем график
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['equity'], label='Стратегия')
    plt.title(f'Equity Curve - {symbol.replace("_", "/")}')
    plt.xlabel('Дата')
    plt.ylabel('Капитал ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Запускаем для обеих монет
run_improved_backtest('BTC_USDT')
run_improved_backtest('ETH_USDT')

print("\nУлучшенный backtest завершён!")