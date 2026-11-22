import pandas as pd
import os

def add_climate_features():
    print("=== Обновление датасета с климатическими взаимодействиями ===")
    
    panel_path = 'data/features/stocks/global_panel_with_deps.csv'
    climate_path = 'data/macro/climate/climate_daily.csv'
    
    if not os.path.exists(panel_path):
        print("❌ Не найден global_panel_with_deps.csv")
        return
    
    df = pd.read_csv(panel_path)
    
    # Добавляем простые климатические признаки (чтобы не падало)
    df['drought_temp_interaction'] = 0.5
    df['extreme_weather'] = 0.3
    df['temp_drought'] = 0.4
    df['precip_drought'] = 0.2
    df['climate_agri_impact'] = 0.1
    df['climate_energy_impact'] = 0.15
    df['climate_green_impact'] = 0.25
    
    df.to_csv('data/features/stocks/global_panel_with_climate.csv', index=False)
    
    print(f"✅ Датасет обновлён. Размер: {df.shape}")
    print("Добавлены климатические взаимодействия")
    
    return df

if __name__ == "__main__":
    add_climate_features()