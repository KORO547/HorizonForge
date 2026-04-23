import pandas as pd
import pandas_ta as ta
import numpy as np
import os

print("=== Универсальная генерация признаков для акций ===\n")

input_dir = 'data/raw/stocks'
output_dir = 'data/features/stocks'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('_1d.csv'):
        ticker = filename.replace('_1d.csv', '')
        print(f"Обрабатываем {ticker}...")

        df = pd.read_csv(f"{input_dir}/{filename}")

        # Приводим данные к правильному формату
        if 'Date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'])
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.set_index('timestamp', inplace=True)

        # Технические индикаторы
        df['rsi_14'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['macd'] = macd.get('MACD_12_26_9', pd.Series(index=df.index))
        df['macd_hist'] = macd.get('MACDh_12_26_9', pd.Series(index=df.index))

        bb = ta.bbands(df['Close'], length=20)
        df['bb_upper'] = bb.iloc[:, 0]
        df['bb_middle'] = bb.iloc[:, 1]
        df['bb_lower'] = bb.iloc[:, 2]

        df['atr_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        for length in [9, 21, 50, 200]:
            df[f'ema_{length}'] = ta.ema(df['Close'], length=length)

        df['price_range'] = (df['High'] - df['Low']) / df['Close']
        df['vol_10'] = df['Close'].pct_change().rolling(10).std()
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

        df['dayofweek'] = df.index.dayofweek
        df['return_1'] = df['Close'].pct_change(1)

        for lag in [1, 3, 5, 10, 20]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)

        # Цель: направление на следующий день
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        df.dropna(inplace=True)

        output_file = f"{output_dir}/{ticker}_features.csv"
        df.to_csv(output_file)

        print(f"✓ {ticker} готово: {len(df)} дней, признаков: {len(df.columns)}\n")

print("Универсальная генерация признаков для акций завершена!")
print(f"Файлы сохранены в {output_dir}/")