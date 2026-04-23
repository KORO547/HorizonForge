import pandas as pd
from datetime import datetime
import os

print("=== Добавление макроэкономических факторов к крипто-признакам ===\n")

# Загружаем макро данные
macro = pd.read_csv('data/macro/macro_data.csv', index_col='DATE', parse_dates=True)

# Загружаем крипто данные (возьмём BTC как пример, потом сделаем для всех)
crypto_files = ['BTC_USDT_features_v6.3.csv', 'ETH_USDT_features_v6.3.csv', 'SOL_USDT_features_v6.3.csv']

for file in crypto_files:
    path = f'data/features/{file}'
    if not os.path.exists(path):
        print(f"Файл {file} не найден, пропускаем")
        continue
        
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    symbol = file.replace('_features_v6.3.csv', '').replace('_', '/')
    print(f"Добавляем макро к {symbol}...")

    # Объединяем по ближайшей дате (forward fill)
    df = df.join(macro, how='left')
    df = df.ffill()                    # заполняем предыдущим значением
    
    # Добавляем простые производные от макро
    df['fed_rate_change'] = df['Fed Interest Rate'].diff()
    df['vix_change'] = df['VIX Volatility Index'].diff()
    df['yield_curve'] = df['10-Year Treasury Yield'] - df['Fed Interest Rate']  # упрощённая кривая доходности
    
    df.reset_index(inplace=True)
    df.to_csv(f'data/features/{file.replace(".csv", "_macro.csv")}', index=False)
    
    print(f"✓ {symbol} + macro готово: {len(df)} строк, признаков: {len(df.columns)}")

print("\nДобавление макро факторов завершено!")
print("Новые файлы сохранены с суффиксом _macro.csv")