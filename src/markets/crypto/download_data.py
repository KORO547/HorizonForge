import ccxt
import pandas as pd
import os
from datetime import datetime

os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/orderbook', exist_ok=True)

exchange = ccxt.binance({
    'enableRateLimit': True,
})

symbols = ['BTC/USDT', 'ETH/USDT']
timeframe = '1h'

print("Начинаем скачивание данных + order book...\n")

for symbol in symbols:
    print(f"Скачиваем {symbol}...")

    # OHLCV
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    filename = f"data/raw/{symbol.replace('/', '_')}_{timeframe}.csv"
    df.to_csv(filename, index=False)
    print(f"✓ OHLCV сохранён: {len(df)} свечей")

    # Order Book snapshot (глубина стакана)
    print(f"Скачиваем order book для {symbol}...")
    try:
        ob = exchange.fetch_order_book(symbol, limit=100)
        
        # Сохраняем bids и asks
        bids = pd.DataFrame(ob['bids'], columns=['price', 'amount'])
        asks = pd.DataFrame(ob['asks'], columns=['price', 'amount'])
        
        bids.to_csv(f"data/orderbook/{symbol.replace('/', '_')}_bids.csv", index=False)
        asks.to_csv(f"data/orderbook/{symbol.replace('/', '_')}_asks.csv", index=False)
        
        print(f"✓ Order book сохранён (bids: {len(bids)}, asks: {len(asks)})")
    except Exception as e:
        print(f"Ошибка при скачивании order book: {e}")

print("\nСкачивание завершено!")