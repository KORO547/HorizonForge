import pandas as pd
from pandas_datareader import data as pdr
import pandas_datareader.data as web
from datetime import datetime
import os

print("=== Загрузка макроэкономических данных (FRED) ===\n")

os.makedirs('data/macro', exist_ok=True)

# Основные макро индикаторы
indicators = {
    'FEDFUNDS': 'Fed Interest Rate',           # Ставка ФРС
    'CPIAUCSL': 'CPI Inflation',               # Инфляция (CPI)
    'UNRATE': 'Unemployment Rate',             # Безработица
    'GDP': 'US GDP Growth',                    # ВВП
    'DGS10': '10-Year Treasury Yield',         # Доходность 10-летних облигаций
    'VIXCLS': 'VIX Volatility Index'           # Индекс страха
}

end = datetime.now()
start = datetime(2024, 1, 1)   # можно взять больше истории

macro_data = pd.DataFrame()

for code, name in indicators.items():
    try:
        df = web.DataReader(code, 'fred', start, end)
        df = df.rename(columns={code: name})
        macro_data = pd.concat([macro_data, df], axis=1)
        print(f"✓ Загружено: {name} ({code})")
    except Exception as e:
        print(f"✗ Ошибка при загрузке {name} ({code}): {e}")

# Сохраняем
macro_data.to_csv('data/macro/macro_data.csv')
print(f"\nМакро данные сохранены в data/macro/macro_data.csv")
print(f"Форма данных: {macro_data.shape}")
print(macro_data.tail(5))