import pandas as pd
import pandas_ta as ta
import numpy as np
import os

print("=== Feature Engineering v6.3 (BTC, ETH, SOL) ===\n")

input_dir = 'data/raw'
ob_history_dir = 'data/orderbook_history'
output_dir = 'data/features'
os.makedirs(output_dir, exist_ok=True)

symbols_to_process = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']

for symbol_clean in symbols_to_process:
    filename = f"{symbol_clean}_1h.csv"
    print(f"Processing {symbol_clean.replace('_', '/')}...")

    df = pd.read_csv(f"{input_dir}/{filename}")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Технические индикаторы
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']

    bb = ta.bbands(df['close'], length=20)
    df['bb_upper'] = bb.iloc[:, 0]
    df['bb_middle'] = bb.iloc[:, 1]
    df['bb_lower'] = bb.iloc[:, 2]

    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    for length in [9, 21, 50]:
        df[f'ema_{length}'] = ta.ema(df['close'], length=length)

    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['vol_10'] = df['close'].pct_change().rolling(10).std()

    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)

    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)

    # Цель: направление через 6 часов
    df['target'] = (df['close'].shift(-6) > df['close']).astype(int)

    # Order Book History
    ob_file = f"{ob_history_dir}/{symbol_clean}_ob_history.csv"
    try:
        ob_df = pd.read_csv(ob_file)
        ob_df['timestamp'] = pd.to_datetime(ob_df['timestamp'])
        ob_df.set_index('timestamp', inplace=True)
        df = df.join(ob_df[['ob_imbalance']], how='left')
        df['ob_imbalance'] = df['ob_imbalance'].ffill().fillna(0)
        df['ob_imbalance_5'] = df['ob_imbalance'].rolling(5).mean()
    except:
        df['ob_imbalance'] = 0.0
        df['ob_imbalance_5'] = 0.0

    df.dropna(inplace=True)
    df.to_csv(f"{output_dir}/{symbol_clean}_features_v6.3.csv")
    print(f"Done {symbol_clean}: {len(df)} rows")

print("\nFeature Engineering v6.3 finished for BTC, ETH, SOL!")