import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

print("=== Улучшенный Walk-Forward Backtest с климатом (консервативный) ===\n")

# Загружаем улучшенную модель
model = joblib.load('models/global_model_climate_enhanced.pkl')
df = pd.read_csv('data/features/stocks/global_panel_with_climate.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"Данные: {len(df)} строк, {df['ticker'].nunique()} акций")

feature_cols = [
    'rsi_14', 'macd', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower',
    'atr_14', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'price_range', 'vol_10', 'volume_ratio',
    'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_10', 'close_lag_20',
    'market_return', 'sector_return',
    'relative_to_market', 'relative_to_sector',
    'market_return_lag1', 'sector_return_lag1',
    'dayofweek',
    'global_temp_anomaly', 'us_temp_anomaly', 
    'precipitation_anomaly', 'drought_index',
    'extreme_heat_days', 'extreme_rain_days',
    'temp_drought', 'precip_drought', 'extreme_weather_score',
    'temp_anomaly_lag1', 'temp_anomaly_lag3', 'temp_anomaly_lag5',
    'drought_lag1', 'drought_lag3', 'drought_lag5',
    'climate_agri_impact', 'climate_energy_impact', 'climate_green_impact'
]

# Консервативные параметры торговли
initial_capital = 10000
commission = 0.0010
min_confidence = 0.66        # повышенный порог
max_positions = 5            # максимум 5 позиций одновременно
risk_per_trade = 0.12        # максимум 12% капитала на одну волну сделок

equity = initial_capital
equity_curve = []

print("Запускаем улучшенный Walk-Forward Backtest...\n")

step = 40
window = 400

for start in tqdm(range(0, len(df) - window, step)):
    end = start + window
    test_start = end
    test_end = min(end + step, len(df))
    
    if test_end - test_start < 20:
        break
    
    test_df = df.iloc[test_start:test_end].copy()
    X_test = test_df[feature_cols]
    test_df = test_df.copy()
    test_df['prob'] = model.predict_proba(X_test)[:, 1]
    
    daily_equity = equity
    open_positions = {}
    
    for ts, group in test_df.groupby('timestamp'):
        # Выход из слабых позиций
        for ticker in list(open_positions.keys()):
            if ticker not in group['ticker'].values:
                continue
            row = group[group['ticker'] == ticker].iloc[0]
            if row['prob'] < 0.53:
                daily_equity *= (1 + row['return_1'] - commission)
                del open_positions[ticker]
        
        # Вход в сильные сигналы
        for _, row in group.iterrows():
            ticker = row['ticker']
            prob = row['prob']
            
            if prob >= min_confidence and len(open_positions) < max_positions and ticker not in open_positions:
                open_positions[ticker] = True
        
        # Расчёт дневной доходности
        daily_ret = 0
        active = len(open_positions)
        if active > 0:
            weight = risk_per_trade / active
            for ticker in open_positions:
                if ticker in group['ticker'].values:
                    ret = group[group['ticker'] == ticker]['return_1'].iloc[0]
                    daily_ret += ret * weight
        
        daily_equity *= (1 + daily_ret)
    
    equity = daily_equity
    equity_curve.append(equity)

# Результаты
equity_curve = pd.Series(equity_curve)
total_return = (equity / initial_capital - 1) * 100
max_dd = ((equity_curve / equity_curve.cummax()) - 1).min() * 100

print("\n" + "="*75)
print("WALK-FORWARD BACKTEST — УЛУЧШЕННАЯ МОДЕЛЬ С КЛИМАТОМ")
print("="*75)
print(f"Начальный капитал      : ${initial_capital:,.0f}")
print(f"Финальный капитал      : ${equity:,.2f}")
print(f"Общая доходность       : {total_return:.2f}%")
print(f"Максимальная просадка  : {max_dd:.2f}%")
print(f"Количество шагов       : {len(equity_curve)}")

# Сохранение и график
os.makedirs('results', exist_ok=True)
equity_curve.to_csv('results/walkforward_climate_enhanced.csv', index=False)

plt.figure(figsize=(12, 6))
plt.plot(equity_curve, label='Global Model with Climate (Enhanced)')
plt.title('Walk-Forward Equity Curve - Enhanced Climate Model')
plt.xlabel('Шаг Walk-Forward')
plt.ylabel('Капитал ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/walkforward_climate_enhanced.png')
plt.show()

print("\nBacktest завершён. График сохранён в results/walkforward_climate_enhanced.png")