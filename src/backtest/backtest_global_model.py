import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

print("=== Backtest Глобальной модели зависимости (186 акций) ===\n")

# Загружаем модель и данные
model_path = 'models/global_dependency_model_v2.pkl'
data_path = 'data/features/stocks/global_panel_with_deps.csv'

model = joblib.load(model_path)
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Загружено {df['ticker'].nunique()} акций, {len(df)} строк")

# Подготовка признаков (точно такие же, как при обучении)
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
df['prob'] = model.predict_proba(X)[:, 1]   # вероятность роста

# Параметры backtest
initial_capital = 10000
commission = 0.0010          # 0.10% комиссия (Binance/Interactive Brokers уровень)
min_confidence = 0.58        # минимальная вероятность для входа

# Симуляция
df = df.sort_values('timestamp')
equity = initial_capital
equity_curve = []
positions = {}

print("Запускаем backtest... (это может занять 1-2 минуты)\n")

for timestamp, group in df.groupby('timestamp'):
    # Закрываем позиции, если вероятность упала
    for ticker in list(positions.keys()):
        if ticker not in group['ticker'].values:
            continue
        prob = group[group['ticker'] == ticker]['prob'].iloc[0]
        if prob < 0.50:   # выходим, если сигнал пропал
            equity *= (1 + group[group['ticker'] == ticker]['return_1'].iloc[0] - commission)
            del positions[ticker]
    
    # Открываем новые позиции
    for _, row in group.iterrows():
        ticker = row['ticker']
        prob = row['prob']
        
        if prob > min_confidence and ticker not in positions:
            # Входим в позицию (равновесный вес)
            position_size = equity * 0.20 / len(group)   # не более 20% на все новые позиции
            positions[ticker] = position_size
    
    # Обновляем equity по открытым позициям
    daily_return = 0
    for ticker in list(positions.keys()):
        if ticker in group['ticker'].values:
            ret = group[group['ticker'] == ticker]['return_1'].iloc[0]
            daily_return += ret * 0.20   # упрощённый вес
    
    equity *= (1 + daily_return)
    equity_curve.append(equity)

# Результаты
equity_curve = pd.Series(equity_curve, index=df['timestamp'].unique())
total_return = (equity / initial_capital - 1) * 100
max_dd = ((equity_curve / equity_curve.cummax()) - 1).min() * 100

print("\n" + "="*70)
print("РЕЗУЛЬТАТЫ BACKTEST ГЛОБАЛЬНОЙ МОДЕЛИ")
print("="*70)
print(f"Начальный капитал     : ${initial_capital:,.0f}")
print(f"Финальный капитал     : ${equity:,.2f}")
print(f"Общая доходность      : {total_return:.2f}%")
print(f"Максимальная просадка : {max_dd:.2f}%")
print(f"Количество дней       : {len(equity_curve)}")

# Сохраняем equity curve
os.makedirs('results', exist_ok=True)
equity_curve.to_csv('results/global_model_equity_curve.csv')

# График
plt.figure(figsize=(12, 6))
plt.plot(equity_curve, label='Global Dependency Model')
plt.title('Equity Curve - Global Dependency Model (186 stocks)')
plt.xlabel('Дата')
plt.ylabel('Капитал ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/global_model_equity_curve.png')
plt.show()

print("\nГрафик сохранён: results/global_model_equity_curve.png")