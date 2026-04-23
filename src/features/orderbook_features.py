import pandas as pd
import os
import numpy as np

print("Начинаем генерацию признаков из order book...\n")

ob_dir = 'data/orderbook'
feature_dir = 'data/features'
os.makedirs(feature_dir, exist_ok=True)

for symbol in ['BTC_USDT', 'ETH_USDT']:
    bids_file = f"{ob_dir}/{symbol}_bids.csv"
    asks_file = f"{ob_dir}/{symbol}_asks.csv"
    
    if not os.path.exists(bids_file) or not os.path.exists(asks_file):
        print(f"Order book для {symbol} не найден")
        continue
    
    bids = pd.read_csv(bids_file)
    asks = pd.read_csv(asks_file)
    
    # Основные метрики имбаланса
    total_bid_volume = bids['amount'].sum()
    total_ask_volume = asks['amount'].sum()
    
    imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume + 1e-8)
    
    # Взвешенный имбаланс (с учётом глубины)
    bid_weighted = (bids['price'] * bids['amount']).sum() / total_bid_volume
    ask_weighted = (asks['price'] * asks['amount']).sum() / total_ask_volume
    
    mid_price = (bids['price'].iloc[0] + asks['price'].iloc[0]) / 2
    weighted_imbalance = (bid_weighted - ask_weighted) / mid_price
    
    # Глубина стакана
    depth_10 = bids['amount'].head(10).sum() - asks['amount'].head(10).sum()
    
    print(f"\n{symbol}:")
    print(f"  Общий имбаланс: {imbalance:.4f}")
    print(f"  Взвешенный имбаланс: {weighted_imbalance:.4f}")
    print(f"  Глубина топ-10: {depth_10:.2f}")
    
    # Здесь мы пока просто выводим. 
    # В следующей версии мы будем добавлять эти признаки к основному датасету.

print("\nАнализ order book завершён.")
print("Следующий шаг — добавить эти метрики как признаки к основным данным.")