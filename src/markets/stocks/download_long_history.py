"""Скачивает длинную дневную историю (10 лет) для всех акций из data/raw/stocks/sp500.

Результат: data/raw/stocks/long/<TICKER>_1d.csv с колонками
timestamp, Open, High, Low, Close, Volume (цены скорректированы на сплиты/дивиденды).

Запуск из корня проекта:
    python src/markets/stocks/download_long_history.py
"""
import os

import pandas as pd
import yfinance as yf

SRC_DIR = "data/raw/stocks/sp500"
OUT_DIR = "data/raw/stocks/long"
PERIOD = "10y"
BATCH = 40

tickers = sorted(f.replace("_1d.csv", "") for f in os.listdir(SRC_DIR) if f.endswith("_1d.csv"))
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Скачиваем {PERIOD} истории для {len(tickers)} тикеров -> {OUT_DIR}")

failed = []
for i in range(0, len(tickers), BATCH):
    batch = tickers[i:i + BATCH]
    data = yf.download(batch, period=PERIOD, interval="1d", auto_adjust=True,
                       group_by="ticker", progress=False, threads=True)
    for t in batch:
        try:
            df = data[t].dropna(how="all") if len(batch) > 1 else data.dropna(how="all")
            if df.empty:
                failed.append(t)
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].reset_index()
            df.rename(columns={"Date": "timestamp"}, inplace=True)
            df.to_csv(f"{OUT_DIR}/{t}_1d.csv", index=False)
        except KeyError:
            failed.append(t)
    print(f"  {min(i + BATCH, len(tickers))}/{len(tickers)}")

print(f"Готово: {len(tickers) - len(failed)} тикеров")
if failed:
    print("Не удалось скачать:", ", ".join(failed))
