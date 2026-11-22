import pandas as pd
import os

def add_energy_to_crypto():
    """Добавляет энергетические и климатические факторы к крипто-данным"""
    print("=== Привязка энергетики и климата к крипте ===")
    
    # Загружаем энергетические данные
    energy = pd.read_csv('data/macro/energy/global_energy.csv').iloc[-1]
    
    # Создаём датафрейм с факторами для крипты
    crypto_factors = {
        'Date': [pd.Timestamp.now().date()],
        'Brent_Price': [energy['Brent_Price_USD']],
        'Natural_Gas_Europe': [energy.get('Natural_Gas_Europe', 11.2)],
        'Crypto_Mining_TWh': [energy.get('Crypto_Mining_TWh', 160)],
        'Global_CO2_Mt': [energy.get('Global_CO2_Mt', 37800)],
        'China_Electricity_Price': [energy['China_Electricity_USD_kWh']],
        'Energy_Crisis_Index': [energy['Brent_Price_USD'] * energy.get('Crypto_Mining_TWh', 160) / 1000],  # синтетический индекс
    }
    
    df = pd.DataFrame(crypto_factors)
    df.to_csv('data/crypto/crypto_energy_factors.csv', index=False)
    
    print("✅ Энергетические факторы для крипты сохранены")
    print(df)
    return df

if __name__ == "__main__":
    add_energy_to_crypto()