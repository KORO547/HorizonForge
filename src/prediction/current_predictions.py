import pandas as pd
import joblib
from datetime import datetime

print("=== Текущие предсказания глобальной модели с климатом ===\n")

# Загружаем модель и последние данные
model = joblib.load('models/global_model_with_climate.pkl')
df = pd.read_csv('data/features/stocks/global_panel_with_climate.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Берём самые последние данные (последний день для каждой акции)
latest_date = df['timestamp'].max()
latest_data = df[df['timestamp'] == latest_date].copy()

print(f"Последняя дата в данных: {latest_date.date()}")
print(f"Количество акций для предсказания: {len(latest_data)}\n")

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
    'drought_temp_interaction', 'extreme_weather'
]

X_latest = latest_data[feature_cols]

# Делаем предсказания
latest_data['prob_up'] = model.predict_proba(X_latest)[:, 1]
latest_data['signal'] = latest_data['prob_up'].apply(lambda x: 'BUY' if x >= 0.63 else 'SELL/HOLD')

# Сортируем по силе сигнала (от strongest BUY к weakest)
latest_data = latest_data.sort_values('prob_up', ascending=False)

# Выводим топ-20 самых сильных сигналов на покупку
print("ТОП-20 САМЫХ СИЛЬНЫХ СИГНАЛОВ НА ПОКУПКУ (prob >= 0.63)\n")
top_buy = latest_data[latest_data['prob_up'] >= 0.63][['ticker', 'prob_up', 'signal']].head(20)
print(top_buy.to_string(index=False))

print("\n" + "="*60)
print("ТОП-10 САМЫХ СЛАБЫХ СИГНАЛОВ (ближайшие к продаже)\n")
bottom = latest_data.tail(10)[['ticker', 'prob_up', 'signal']]
print(bottom.to_string(index=False))

# Сохраняем все предсказания
latest_data[['ticker', 'prob_up', 'signal']].to_csv('results/current_predictions.csv', index=False)
print("\nВсе предсказания сохранены в: results/current_predictions.csv")

# Краткая статистика
buy_count = (latest_data['prob_up'] >= 0.63).sum()
print(f"\nСейчас модель рекомендует покупать: {buy_count} акций из {len(latest_data)}")
print(f"Средняя вероятность роста: {latest_data['prob_up'].mean():.4f}")