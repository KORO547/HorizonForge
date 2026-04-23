import pandas as pd
import os
from tqdm import tqdm

print("=== Создание Global Panel Dataset (все 158 акций вместе) ===\n")

input_dir = 'data/features/stocks/sp500'
output_file = 'data/features/stocks/global_panel.csv'

files = [f for f in os.listdir(input_dir) if f.endswith('_features.csv')]

all_data = []
print(f"Объединяем {len(files)} акций...\n")

for filename in tqdm(files):
    ticker = filename.replace('_features.csv', '')
    df = pd.read_csv(f"{input_dir}/{filename}")
    df['ticker'] = ticker
    all_data.append(df)

# Объединяем всё в один большой DataFrame
panel = pd.concat(all_data, ignore_index=True)

# Сортируем по времени и тикеру
panel['timestamp'] = pd.to_datetime(panel['timestamp'])
panel = panel.sort_values(['ticker', 'timestamp']).reset_index(drop=True)

print(f"\nГотово! Размер панельного датасета: {panel.shape}")
print(f"Акций: {panel['ticker'].nunique()}")
print(f"Строк: {len(panel)}")
print(f"Признаков: {panel.shape[1]}")

# Сохраняем
panel.to_csv(output_file, index=False)
print(f"\nGlobal Panel Dataset сохранён в: {output_file}")