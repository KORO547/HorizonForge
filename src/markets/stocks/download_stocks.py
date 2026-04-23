import yfinance as yf
import pandas as pd
import os
from datetime import datetime

print("=== Скачивание данных по 10 секторам акций ===\n")

tickers = [
    # 1. Технологии
    'AAPL', 'MSFT', 'NVDA', 'GOOGL',
    # 2. Полупроводники
    'AMD', 'TSM', 'AVGO', 'ASML',
    # 3. Индексы
    'SPY', 'QQQ',
    # 4. Финансы
    'JPM', 'BAC', 'V', 'MA',
    # 5. Здравоохранение
    'JNJ', 'UNH', 'LLY', 'PFE',
    # 6. Энергетика
    'XOM', 'CVX',
    # 7. Потребительский
    'AMZN', 'TSLA', 'HD', 'NKE',
    # 8. Промышленность
    'CAT', 'GE', 'HON',
    # 9. Коммунальные услуги
    'NEE', 'DUK',
    # 10. Недвижимость
    'AMT', 'PLD'
]

os.makedirs('data/raw/stocks', exist_ok=True)

for ticker in tickers:
    print(f"Скачиваем {ticker}...")

    df = yf.download(ticker, period="2y", interval="1d")
    
    if df.empty:
        print(f"✗ Не удалось скачать {ticker}")
        continue
    
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'timestamp'}, inplace=True)
    
    filename = f"{ticker}_1d.csv"
    df.to_csv(f'data/raw/stocks/{filename}', index=False)
    
    print(f"✓ {ticker} скачано: {len(df)} дней")

print("\nСкачивание данных по 10 секторам завершено!")
print(f"Всего тикеров: {len(tickers)}")