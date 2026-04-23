import ccxt
import pandas as pd
import os
from datetime import datetime

print("=== Download Data + Order Book History (BTC, ETH, SOL) ===\n")

exchange = ccxt.binance({'enableRateLimit': True})

symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
timeframe = '1h'
limit = 1000

os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/orderbook_history', exist_ok=True)

for symbol in symbols:
    print(f"Downloading {symbol}...")

    # OHLCV
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.to_csv(f'data/raw/{symbol.replace("/", "_")}_1h.csv', index=False)

    # Order Book + Imbalance
    ob = exchange.fetch_order_book(symbol, limit=50)

    bids_vol = sum(amount for price, amount in ob['bids'])
    asks_vol = sum(amount for price, amount in ob['asks'])
    imbalance = (bids_vol - asks_vol) / (bids_vol + asks_vol) if (bids_vol + asks_vol) > 0 else 0

    history_file = f'data/orderbook_history/{symbol.replace("/", "_")}_ob_history.csv'

    new_row = pd.DataFrame([{
        'timestamp': datetime.now(),
        'ob_imbalance': imbalance,
        'bids_vol': bids_vol,
        'asks_vol': asks_vol
    }])

    if os.path.exists(history_file):
        history = pd.read_csv(history_file)
        history = pd.concat([history, new_row], ignore_index=True)
    else:
        history = new_row

    history.to_csv(history_file, index=False)

    print(f"DONE {symbol} | Imbalance: {imbalance:.4f}")

print("\nDownload completed for BTC, ETH, SOL!")
