import pandas as pd
import os
from datetime import datetime

def create_energy_data():
    os.makedirs('data/macro/energy', exist_ok=True)
    
    print("=== Полная энергетическая + климатическая карта (с CO₂) ===")
    
    data = {
        'Year': [2024, 2025],
        'China_TWh': [9200, 9500],
        'USA_TWh': [4150, 4200],
        'EU_TWh': [2550, 2500],
        'India_TWh': [1600, 1720],
        'Russia_TWh': [1070, 1060],
        'Africa_TWh': [880, 920],
        'Latin_America_TWh': [1130, 1170],
        'Australia_TWh': [290, 300],
        'Canada_TWh': [660, 670],
        
        'Brent_Price_USD': [78, 85],
        'Natural_Gas_USA': [3.2, 4.1],
        'Natural_Gas_Europe': [9.5, 11.2],
        'Natural_Gas_Asia': [12.8, 13.5],
        
        'China_Electricity_USD_kWh': [0.112, 0.115],
        'USA_Electricity_USD_kWh': [0.135, 0.140],
        'EU_Electricity_USD_kWh': [0.218, 0.210],
        'India_Electricity_USD_kWh': [0.108, 0.110],
        'Russia_Electricity_USD_kWh': [0.078, 0.080],
        
        # === Выбросы CO₂ (млн тонн) ===
        'Global_CO2_Mt': [37200, 37800],
        'China_CO2_Mt': [11800, 11950],
        'USA_CO2_Mt': [4800, 4750],
        'EU_CO2_Mt': [2800, 2650],
        'India_CO2_Mt': [2600, 2750],
        'Russia_CO2_Mt': [1800, 1780],
        
        # === Выбросы CO₂ по отраслям (глобально, % от общего) ===
        'Energy_Sector_CO2_pct': [73, 72],
        'Industry_CO2_pct': [24, 25],           # включая производство бетона
        'Agriculture_CO2_pct': [12, 12],
        'Transport_CO2_pct': [14, 14],
        'Buildings_CO2_pct': [6, 6],
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/macro/energy/global_energy.csv', index=False)
    
    print("✅ Добавлены данные по CO₂ и отраслям")
    return df

if __name__ == "__main__":
    create_energy_data()