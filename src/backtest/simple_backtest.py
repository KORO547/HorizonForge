import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("=== Простой Backtest стратегии ===\n")

# Параметры торговли
commission = 0.00075   # 0.075% комиссия (типичная для Binance)
initial_capital = 10000  # стартовый капитал $10,000

def run_backtest(symbol):
    # Загружаем данные с признаками
    df = pd.read_csv(f'data/features/{symbol}_1h_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Загружаем предсказания из последней модели (пока используем target_dir как "идеальное" предсказание для теста)
    # В реальности здесь будут предсказания модели
    df['predicted_dir'] = df['target_dir'].shift(1)   # сдвиг, чтобы не использовать будущее
    
    # Простая стратегия: покупаем, если модель говорит "вверх", продаём при "вниз"
    df['position'] = 0
    df.loc[df['predicted_dir'] == 1, 'position'] = 1   # long
    df.loc[df['predicted_dir'] == 0, 'position'] = -1  # short (или 0, если только long)
    
    # Доходность стратегии
    df['strategy_return'] = df['position'].shift(1) * df['return_1']
    
    # Учитываем комиссию при смене позиции
    df['trade'] = df['position'].diff().abs()
    df['strategy_return'] = df['strategy_return'] - df['trade'] * commission
    
    # Кумулятивная доходность
    df['cum_return'] = (1 + df['strategy_return']).cumprod()
    df['cum_capital'] = initial_capital * df['cum_return']
    
    # Метрики
    total_return = df['cum_return'].iloc[-1] - 1
    num_trades = df['trade'].sum()
    sharpe = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(24*365) if df['strategy_return'].std() != 0 else 0
    
    print(f"\n=== Результаты для {symbol.replace('_', '/')} ===")
    print(f"Итоговая доходность: {total_return*100:.2f}%")
    print(f"Финальный капитал: ${df['cum_capital'].iloc[-1]:.2f}")
    print(f"Количество сделок: {num_trades:.0f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Макс. просадка: {((df['cum_capital'] / df['cum_capital'].cummax()) - 1).min()*100:.2f}%")
    
    # Сохраняем equity curve
    os.makedirs('results', exist_ok=True)
    df[['timestamp', 'cum_capital']].to_csv(f'results/equity_{symbol}.csv', index=False)
    
    # Строим график
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['cum_capital'], label='Стратегия')
    plt.title(f'Equity Curve - {symbol.replace("_", "/")}')
    plt.xlabel('Дата')
    plt.ylabel('Капитал ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Запускаем backtest
run_backtest('BTC_USDT')
run_backtest('ETH_USDT')

print("\nBacktest завершён!")
print("Результаты сохранены в папке 'results/'")