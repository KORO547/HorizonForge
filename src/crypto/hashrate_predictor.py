import pandas as pd
import os
from datetime import datetime
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore")

def predict_hashrate():
    print("=== Прогнозирование хешрейта Bitcoin ===")
    
    # Загружаем энергетические факторы
    energy = pd.read_csv('data/macro/energy/global_energy.csv').iloc[-1]
    
    # Синтетическая история + текущие данные
    data = {
        'Date': pd.date_range(end=datetime.now().date(), periods=60),
        'Brent_Price': [75 + i*0.15 for i in range(60)],
        'Electricity_Price_Avg': [0.11 + i*0.0005 for i in range(60)],
        'Crypto_Mining_TWh': [120 + i*0.7 for i in range(60)],
        'Hashrate_EH_s': [450 + i*8 for i in range(60)]  # цель — хешрейт в EH/s
    }
    
    df = pd.DataFrame(data)
    
    # Признаки
    X = df[['Brent_Price', 'Electricity_Price_Avg', 'Crypto_Mining_TWh']]
    y = df['Hashrate_EH_s']
    
    # Обучение
    model = LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    
    # Прогноз на следующий период
    latest = X.iloc[-1:].copy()
    predicted_hashrate = model.predict(latest)[0]
    
    print(f"✅ Прогноз хешрейта Bitcoin:")
    print(f"Текущий (примерно): {y.iloc[-1]:.0f} EH/s")
    print(f"Прогноз на следующий период: {predicted_hashrate:.0f} EH/s")
    print(f"Изменение: {'↑' if predicted_hashrate > y.iloc[-1] else '↓'}")
    
    # Сохранение
    os.makedirs('models', exist_ok=True)
    # Здесь можно сохранить модель
    
    return predicted_hashrate

if __name__ == "__main__":
    predict_hashrate()