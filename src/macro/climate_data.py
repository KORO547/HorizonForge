import pandas as pd
import numpy as np
import os
from datetime import datetime

print("=== Загрузка климатических и погодных данных ===\n")

os.makedirs('data/macro/climate', exist_ok=True)

# Создаём диапазон дат
dates = pd.date_range(start='2024-01-01', end='2026-04-22', freq='D')

np.random.seed(42)

# Генерируем реалистичные климатические данные
climate_df = pd.DataFrame({
    'timestamp': dates,
    
    # Температурные аномалии (°C)
    'global_temp_anomaly': np.random.normal(1.25, 0.45, len(dates)),
    'us_temp_anomaly': np.random.normal(1.1, 0.55, len(dates)),
    
    # Аномалия осадков (стандартные отклонения)
    'precipitation_anomaly': np.random.normal(0, 1.1, len(dates)),
    
    # Индекс засухи (Palmer Drought Severity Index)
    'drought_index': np.random.normal(2.8, 1.8, len(dates)).clip(0, 7),
    
    # Экстремальные погодные события (0-1)
    'extreme_heat_days': np.random.binomial(1, 0.12, len(dates)),
    'extreme_rain_days': np.random.binomial(1, 0.08, len(dates)),
})

climate_df.set_index('timestamp', inplace=True)

# Добавляем сезон
climate_df['month'] = climate_df.index.month
climate_df['season'] = climate_df['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
}).fillna('Other')

print(f"Климатические данные успешно созданы: {len(climate_df)} дней")

print("\nДобавленные показатели:")
print("- global_temp_anomaly (глобальная температурная аномалия)")
print("- us_temp_anomaly (аномалия температуры по США)")
print("- precipitation_anomaly (аномалия осадков)")
print("- drought_index (индекс засухи)")
print("- extreme_heat_days и extreme_rain_days (экстремальная погода)")

# Сохраняем
climate_df.to_csv('data/macro/climate/climate_daily.csv')
print("\nФайл сохранён: data/macro/climate/climate_daily.csv")

print("\nГотовы к интеграции климатических данных с акциями.")