import pandas as pd
import os
from datetime import datetime

def add_macro_to_crypto():
    print("=== Добавление макро и энергетических факторов к крипте ===")
    
    # Загружаем энергетические данные
    energy = pd.read_csv('data/macro/energy/global_energy.csv').iloc[-1]
    
    # Создаём факторы для крипты
    factors = {
        'Date': [datetime.now().date()],
        'Brent_Price': [energy['Brent_Price_USD']],
        'Gas_Europe': [energy.get('Natural_Gas_Europe', 11.2)],
        'Crypto_Mining_TWh': [energy.get('Crypto_Mining_TWh', 160)],
        'Global_CO2': [energy.get('Global_CO2_Mt', 37800)],
        'China_Electricity_Price': [energy['China_Electricity_USD_kWh']],
        'Energy_Crisis_Index': [energy['Brent_Price_USD'] * 0.8 + energy.get('Natural_Gas_Europe', 11.2) * 0.2],  # синтетический
        'Mining_Difficulty_Index': [energy.get('Crypto_Mining_TWh', 160) / 100]  # условный
    }
    
    df = pd.DataFrame(factors)
    os.makedirs('data/crypto', exist_ok=True)
    df.to_csv('data/crypto/crypto_macro_factors.csv', index=False)
    
    print("✅ Макро-факторы для крипты добавлены")
    print(df)
    return df

if __name__ == "__main__":
    add_macro_to_crypto()