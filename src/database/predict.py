from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

engine = create_engine(
    "postgresql+psycopg2://horizonforge:horizon123@localhost:5432/horizonforge"
)

def simple_forecast(asset_id, asset_name, days=7):
    with engine.connect() as conn:
        df = pd.read_sql(f"""
            SELECT time, price 
            FROM market_prices 
            WHERE asset_id = {asset_id} 
            ORDER BY time DESC 
            LIMIT 60
        """, conn)
    
    if len(df) < 10:
        return None
    
    df = df.sort_values('time')
    prices = df['price'].values
    
    # Простая линейная регрессия + небольшая случайность
    x = np.arange(len(prices))
    slope, intercept = np.polyfit(x, prices, 1)
    
    last_price = prices[-1]
    forecast = []
    current = last_price
    
    for i in range(days):
        current = current * (1 + slope / last_price * 0.7) + np.random.normal(0, last_price * 0.003)
        forecast.append(round(float(current), 2))
    
    return {
        "asset": asset_name,
        "current_price": round(float(last_price), 2),
        "forecast_7d": forecast[-1],
        "forecast_prices": forecast,
        "trend": "growth" if slope > 0 else "decline"
    }

def generate_forecasts():
    print("🔮 Генерируем прогнозы...")
    
    forecasts = []
    for asset_id, name in [(1, "Bitcoin"), (2, "Ethereum"), (3, "Solana")]:
        pred = simple_forecast(asset_id, name)
        if pred:
            forecasts.append(pred)
            print(f"✅ {name}: ${pred['current_price']} → ${pred['forecast_7d']} (7 дней)")
    
    return forecasts

if __name__ == "__main__":
    generate_forecasts()