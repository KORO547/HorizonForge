import pandas as pd
import numpy as np
from datetime import datetime
import os

print("=== Новостной сентимент (упрощённая версия без API) ===\n")

os.makedirs('data/news', exist_ok=True)

# Симулируем сентимент на основе случайного, но воспроизводимого значения + тренда
np.random.seed(42)

assets = ['BTC', 'ETH', 'SOL', 'MACRO']
dates = pd.date_range(end=datetime.now(), periods=200, freq='H')  # последние 200 часов

for asset in assets:
    sentiment_data = []
    
    for date in dates:
        # Базовый сентимент + небольшой шум + тренд
        base_sentiment = 0.1 if asset in ['BTC', 'ETH'] else 0.0
        noise = np.random.normal(0, 0.3)
        trend = np.sin(date.timestamp() / 86400) * 0.2   # лёгкий суточный тренд
        
        polarity = np.clip(base_sentiment + noise + trend, -1.0, 1.0)
        
        sentiment_data.append({
            'timestamp': date,
            'asset': asset,
            'sentiment': polarity,
            'title': f"Simulated news for {asset} at {date.strftime('%Y-%m-%d %H:%M')}"
        })
    
    df = pd.DataFrame(sentiment_data)
    df.to_csv(f'data/news/{asset}_news_sentiment.csv', index=False)
    
    avg_sentiment = df['sentiment'].mean()
    print(f"✓ {asset}: {len(df)} записей, средний сентимент: {avg_sentiment:.3f}")

print("\nУпрощённый новостной сентимент сгенерирован!")
print("Файлы сохранены в data/news/")