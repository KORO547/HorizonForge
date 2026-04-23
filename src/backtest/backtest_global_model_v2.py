import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

print("=== Улучшенный Backtest Глобальной модели (v2) ===\n")

# Параметры
initial_capital = 10000
commission = 0.0010          # 0.10%
min_confidence = 0.62        # повысили порог входа
max_positions = 8            # максимум одновременно открытых позиций
risk_per_trade = 0.15        # максимум 15% капитала на одну позицию
stop_loss = -0.08            # стоп-лосс -8%

# Загрузка
model = joblib.load('models/global_dependency_model_v2.pkl')
df = pd.read_csv('data/features/stocks/global_panel_with_deps.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

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

X = df[feature_cols]
df['prob'] = model.predict_proba(X)[:, 1]

# Backtest
df = df.sort_values('timestamp').reset_index(drop=True)
equity = initial_capital
equity_curve = []
open_positions = {}   # ticker -> entry_price

print("Запускаем улучшенный backtest...\n")

for ts, group in df.groupby('timestamp'):
    current_prices = group.set_index('ticker')['Close'] if 'Close' in group.columns else None
    
    # Проверяем стоп-лосс и выходим из слабых позиций
    to_close = []
    for ticker, entry_price in open_positions.items():
        if ticker in group['ticker'].values:
            current_price = group[group['ticker'] == ticker]['Close'].iloc[0]
            ret = (current_price / entry_price) - 1
            prob = group[group['ticker'] == ticker]['prob'].iloc[0]
            
            if ret < stop_loss or prob < 0.48:
                equity *= (1 + ret - commission)
                to_close.append(ticker)
    
    for ticker in to_close:
        del open_positions[ticker]
    
    # Открываем новые позиции
    candidates = group[group['prob'] >= min_confidence].sort_values('prob', ascending=False)
    
    for _, row in candidates.iterrows():
        ticker = row['ticker']
        if ticker in open_positions or len(open_positions) >= max_positions:
            continue
            
        position_size = equity * risk_per_trade
        open_positions[ticker] = row['Close']   # запоминаем цену входа
    
    # Обновляем equity по открытым позициям
    daily_pnl = 0
    for ticker, entry_price in list(open_positions.items()):
        if ticker in group['ticker'].values:
            current_price = group[group['ticker'] == ticker]['Close'].iloc[0]
            ret = (current_price / entry_price) - 1
            daily_pnl += ret * risk_per_trade
    
    equity *= (1 + daily_pnl)
    equity_curve.append(equity)

# Результаты
equity_curve = pd.Series(equity_curve, index=df['timestamp'].unique()[:len(equity_curve)])
total_return = (equity / initial_capital - 1) * 100
max_dd = ((equity_curve / equity_curve.cummax()) - 1).min() * 100

print("\n" + "="*75)
print("РЕЗУЛЬТАТЫ УЛУЧШЕННОГО BACKTEST")
print("="*75)
print(f"Начальный капитал      : ${initial_capital:,.0f}")
print(f"Финальный капитал      : ${equity:,.2f}")
print(f"Общая доходность       : {total_return:.2f}%")
print(f"Максимальная просадка  : {max_dd:.2f}%")
print(f"Макс. одновременно открытых позиций : {max_positions}")
print(f"Минимальная уверенность входа      : {min_confidence}")

# Сохранение
os.makedirs('results', exist_ok=True)
equity_curve.to_csv('results/global_model_equity_curve_v2.csv')

# График
plt.figure(figsize=(12, 6))
plt.plot(equity_curve, label='Global Dependency Model v2')
plt.title('Equity Curve - Global Dependency Model v2 (186 stocks)')
plt.xlabel('Дата')
plt.ylabel('Капитал ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/global_model_equity_curve_v2.png')
plt.show()

print("\nГрафик сохранён: results/global_model_equity_curve_v2.png")