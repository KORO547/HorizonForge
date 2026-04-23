import pandas as pd
import numpy as np

print("=== Обновление датасета с климатическими взаимодействиями ===\n")

stocks_df = pd.read_csv('data/features/stocks/global_panel_with_deps.csv')
stocks_df['timestamp'] = pd.to_datetime(stocks_df['timestamp'])

climate_df = pd.read_csv('data/macro/climate/climate_daily.csv')
climate_df['timestamp'] = pd.to_datetime(climate_df['timestamp'])

# Присоединяем климат
merged_df = stocks_df.merge(climate_df, on='timestamp', how='left')

# Добавляем взаимодействия
merged_df['temp_drought'] = merged_df['global_temp_anomaly'] * merged_df['drought_index']
merged_df['precip_drought'] = merged_df['precipitation_anomaly'] * merged_df['drought_index']
merged_df['extreme_weather_score'] = merged_df['extreme_heat_days'] + merged_df['extreme_rain_days']

# Лаггированные климатические признаки
for lag in [1, 3, 5]:
    merged_df[f'temp_anomaly_lag{lag}'] = merged_df.groupby('ticker')['global_temp_anomaly'].shift(lag)
    merged_df[f'drought_lag{lag}'] = merged_df.groupby('ticker')['drought_index'].shift(lag)
    merged_df[f'precip_lag{lag}'] = merged_df.groupby('ticker')['precipitation_anomaly'].shift(lag)

# Секторные взаимодействия
merged_df['is_agri'] = merged_df['ticker'].isin(['CTVA','MOS','CF','NTR','ADM','BG','FMC','AGCO']).astype(int)
merged_df['is_energy'] = merged_df['ticker'].isin(['XOM','CVX','COP','SLB','EOG']).astype(int)
merged_df['is_green'] = merged_df['ticker'].isin(['ENPH','FSLR','BEPC','PLUG','BLDP','SEDG','RUN']).astype(int)

merged_df['climate_agri_impact'] = merged_df['is_agri'] * (merged_df['drought_index'] + merged_df['global_temp_anomaly'])
merged_df['climate_energy_impact'] = merged_df['is_energy'] * merged_df['global_temp_anomaly']
merged_df['climate_green_impact'] = merged_df['is_green'] * merged_df['global_temp_anomaly']

print(f"Финальный размер датасета: {merged_df.shape}")

merged_df.to_csv('data/features/stocks/global_panel_with_climate.csv', index=False)
print("Обновлённый датасет с климатическими взаимодействиями сохранён!")