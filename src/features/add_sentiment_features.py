import pandas as pd
import os
from datetime import datetime

print("=== Добавление новостного сентимента к признакам ===\n")

news_dir = 'data/news'
features_dir = 'data/features'
output_dir = 'data/features'
os.makedirs(output_dir, exist_ok=True)

assets_map = {
    'BTC_USDT': 'BTC',
    'ETH_USDT': 'ETH',
    'SOL_USDT': 'SOL'
}

for symbol_clean, news_asset in assets_map.items():
    feature_file = f"{features_dir}/{symbol_clean}_features_v6.3_macro.csv"
    if not os.path.exists(feature_file):
        print(f"Файл признаков для {symbol_clean} не найден")
        continue

    print(f"Добавляем сентимент для {symbol_clean.replace('_', '/')}...")

    df = pd.read_csv(feature_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Загружаем сентимент
    news_file = f"{news_dir}/{news_asset}_news_sentiment.csv"
    try:
        sentiment_df = pd.read_csv(news_file)
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
        sentiment_df.set_index('timestamp', inplace=True)
        
        # Объединяем по ближайшему времени
        df = df.join(sentiment_df[['sentiment']], how='left')
        df['sentiment'] = df['sentiment'].ffill().fillna(0)
        
        # Делаем полезные признаки из сентимента
        df['sentiment_5'] = df['sentiment'].rolling(5).mean()
        df['sentiment_10'] = df['sentiment'].rolling(10).mean()
        df['sentiment_change'] = df['sentiment'].diff()
        
        print(f"✓ Сентимент добавлен для {symbol_clean}")
    except Exception as e:
        print(f"Не удалось добавить сентимент для {symbol_clean}: {e}")
        df['sentiment'] = 0.0
        df['sentiment_5'] = 0.0
        df['sentiment_10'] = 0.0
        df['sentiment_change'] = 0.0

    df.reset_index(inplace=True)
    output_file = f"{output_dir}/{symbol_clean}_features_v6.3_macro_sentiment.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Готово: {len(df)} строк, признаков: {len(df.columns)}\n")

print("Добавление сентимента завершено!")
print("Новые файлы сохранены с суффиксом _macro_sentiment.csv")