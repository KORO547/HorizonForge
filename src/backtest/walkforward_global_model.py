import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

print("=== Walk-Forward Backtest Глобальной модели (честная проверка) ===\n")

# Загружаем данные и модель
df = pd.read_csv('data/features/stocks/global_panel_with_deps.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

model = joblib.load('models/global_dependency_model_v2.pkl')

feature_cols = [
    'rsi_14', 'macd', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower',
    'atr_14', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'price_range', 'vol_10', 'volume_ratio',
    'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_10', 'close_lag_20',
    'market_return', 'sector_return',
    'relative_to_market', 'relative_to_sector',
    'market_return_lag1', 'sector_return_lag1',
    'dayofweek'
]

print(f"Данные загружены: {len(df)} строк, {df['ticker'].nunique()} акций")

# Параметры walk-forward и торговли
initial_capital = 10000
commission = 0.0010          # 0.10%
min_confidence = 0.63        # повышаем порог
max_positions = 8            # максимум одновременно открытых позиций
window_size = 400            # размер окна обучения (примерно 1.5 года)
step = 60                    # шаг вперёд (примерно 3 месяца)

equity = initial_capital
equity_curve = []
trade_log = []

print("Запускаем Walk-Forward Backtest...\n")

for start in tqdm(range(0, len(df) - window_size, step)):
    end = start + window_size
    test_start = end
    test_end = min(end + step, len(df))
    
    if test_end - test_start < 30:
        break
    
    train_df = df.iloc[start:end]
    test_df = df.iloc[test_start:test_end]
    
    # Предсказания на тестовом периоде
    X_test = test_df[feature_cols]
    test_df = test_df.copy()
    test_df['prob'] = model.predict_proba(X_test)[:, 1]
    
    # Торговля в тестовом окне
    daily_equity = equity
    open_positions = {}
    
    for ts, group in test_df.groupby('timestamp'):
        # Выход из позиций
        for ticker in list(open_positions.keys()):
            if ticker not in group['ticker'].values:
                continue
            row = group[group['ticker'] == ticker].iloc[0]
            if row['prob'] < 0.50:
                daily_equity *= (1 + row['return_1'] - commission)
                del open_positions[ticker]
        
        # Вход в новые позиции
        for _, row in group.iterrows():
            ticker = row['ticker']
            prob = row['prob']
            
            if prob >= min_confidence and len(open_positions) < max_positions and ticker not in open_positions:
                open_positions[ticker] = True
        
        # Обновление equity по открытым позициям
        daily_ret = 0
        for ticker in open_positions:
            if ticker in group['ticker'].values:
                ret = group[group['ticker'] == ticker]['return_1'].iloc[0]
                daily_ret += ret * (1.0 / max_positions)
        
        daily_equity *= (1 + daily_ret - commission * 0.1)  # небольшая комиссия за удержание
    
    equity = daily_equity
    equity_curve.append(equity)

# Результаты
equity_curve = pd.Series(equity_curve)
total_return = (equity / initial_capital - 1) * 100
max_dd = ((equity_curve / equity_curve.cummax()) - 1).min() * 100

print("\n" + "="*70)
print("WALK-FORWARD BACKTEST РЕЗУЛЬТАТЫ")
print("="*70)
print(f"Начальный капитал     : ${initial_capital:,.0f}")
print(f"Финальный капитал     : ${equity:,.2f}")
print(f"Общая доходность      : {total_return:.2f}%")
print(f"Максимальная просадка : {max_dd:.2f}%")
print(f"Количество шагов      : {len(equity_curve)}")

# Сохраняем и рисуем график
os.makedirs('results', exist_ok=True)
equity_curve.to_csv('results/walkforward_global_equity.csv', index=False)

plt.figure(figsize=(12, 6))
plt.plot(equity_curve, label='Global Dependency Model - Walk Forward')
plt.title('Walk-Forward Equity Curve - Global Dependency Model (186 stocks)')
plt.xlabel('Шаг Walk-Forward')
plt.ylabel('Капитал ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/walkforward_global_equity.png')
plt.show()

print("\nWalk-Forward backtest завершён.")
print("График сохранён: results/walkforward_global_equity.png")