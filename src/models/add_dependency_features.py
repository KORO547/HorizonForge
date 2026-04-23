import pandas as pd
import numpy as np

print("=== Добавление признаков зависимостей (Market + Sector Factors) ===\n")

# Загружаем панель
df = pd.read_csv('data/features/stocks/global_panel.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Исходный размер: {df.shape}")

# 1. Рыночный фактор (Market Return)
market = df.groupby('timestamp')['return_1'].mean().reset_index()
market.rename(columns={'return_1': 'market_return'}, inplace=True)

df = df.merge(market, on='timestamp', how='left')

# 2. Секторный фактор
# Пока просто группируем по первым буквам тикера как грубый сектор (можно улучшить позже)
df['sector'] = df['ticker'].str[:2]   # грубая группировка

sector_return = df.groupby(['timestamp', 'sector'])['return_1'].mean().reset_index()
sector_return.rename(columns={'return_1': 'sector_return'}, inplace=True)

df = df.merge(sector_return, on=['timestamp', 'sector'], how='left')

# 3. Относительная сила акции
df['relative_to_market'] = df['return_1'] - df['market_return']
df['relative_to_sector'] = df['return_1'] - df['sector_return']

# 4. Лаги рыночных и секторных факторов
df['market_return_lag1'] = df.groupby('ticker')['market_return'].shift(1)
df['sector_return_lag1'] = df.groupby('ticker')['sector_return'].shift(1)

print(f"\nФинальный размер датасета: {df.shape}")
print("Добавленные признаки зависимостей:")
print("- market_return")
print("- sector_return")
print("- relative_to_market")
print("- relative_to_sector")
print("- market_return_lag1, sector_return_lag1")

# Сохраняем
df.to_csv('data/features/stocks/global_panel_with_deps.csv', index=False)
print("\nДатасет с зависимостями сохранён в: data/features/stocks/global_panel_with_deps.csv")