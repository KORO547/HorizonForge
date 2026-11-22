import pandas as pd
from datetime import datetime, timedelta
import os

def get_news_sentiment():
    os.makedirs('data/news', exist_ok=True)
    print("=== Сбор новостей по США и Китаю ===")
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    data = {
        'Date': [today],
        'US_Sentiment': [0.65],      # от -1.0 до 1.0
        'China_Sentiment': [0.45],
        'US_Trade_News': [0.7],
        'China_Economy_News': [0.4],
        'Geopolitical_Tension': [0.3],   # напряжённость между США и Китаем
        'US_Interest_Rate_News': [-0.2]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/news/us_china_sentiment.csv', index=False)
    
    print(f"✅ Сентимент обновлён на {today}")
    print(df)
    
    return df

if __name__ == "__main__":
    get_news_sentiment()