import pandas as pd
import os
from datetime import datetime

def integrate_macro_to_crypto():
    print("=== Интеграция макро и энергетических факторов в крипто-прогнозирование ===")
    
    # Загружаем энергетические данные
    energy_path = 'data/macro/energy/global_energy.csv'
    if not os.path.exists(energy_path):
        print("❌ Нет энергетических данных")
        return None
    
    energy = pd.read_csv(energy_path).iloc[-1]
    
    # Создаём объединённый датасет для крипты
    crypto_macro = {
        'Date': [datetime.now().date()],
        'BTC_Price': [94500],           # заглушка, в будущем — реальные данные
        'ETH_Price': [2780],
        'SOL_Price': [178],
        
        # Макро-факторы
        'Brent_Price': [energy['Brent_Price_USD']],
        'Gas_Europe': [energy.get('Natural_Gas_Europe', 11.2)],
        'Crypto_Mining_Global_TWh': [energy.get('Crypto_Mining_TWh', 160)],
        'Global_CO2_Mt': [energy.get('Global_CO2_Mt', 37800)],
        'China_Electricity_Price': [energy['China_Electricity_USD_kWh']],
        'Energy_Crisis_Index': [round(energy['Brent_Price_USD'] * 0.65 + energy.get('Natural_Gas_Europe', 11.2) * 0.35, 2)],
        
        # Целевые переменные (в будущем — реальные)
        'BTC_Target': [1],   # 1 = рост, 0 = падение
        'ETH_Target': [0],
        'SOL_Target': [1]
    }
    
    df = pd.DataFrame(crypto_macro)
    
    os.makedirs('data/crypto', exist_ok=True)
    df.to_csv('data/crypto/crypto_with_macro.csv', index=False)
    
    print("✅ Крипто + Макро + Энергетика успешно объединены")
    print(df)
    
    return df

if __name__ == "__main__":
    integrate_macro_to_crypto()