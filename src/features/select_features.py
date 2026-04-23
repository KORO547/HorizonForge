import pandas as pd
import os

print("=== Отбор лучших признаков ===\n")

top_features = {}

for symbol_file in ['BTC_USDT_1h_features.csv', 'ETH_USDT_1h_features.csv']:
    file_path = f'data/features/{symbol_file}'
    df = pd.read_csv(file_path)
    
    feature_cols = [col for col in df.columns 
                    if col not in ['timestamp', 'target', 'target_dir']]
    
    symbol_name = symbol_file.replace('_1h_features.csv', '').replace('_', '/')
    
    # Корреляция
    corr = df[feature_cols + ['target']].corr()['target'].abs()
    corr_sorted = corr.sort_values(ascending=False)
    
    # Берём топ-35 признаков (исключая target)
    selected = corr_sorted.head(36).index.tolist()   # 35 + target
    selected = [col for col in selected if col != 'target']
    
    top_features[symbol_name] = selected[:35]   # оставляем ровно 35
    
    print(f"\n{symbol_name} — отобрано 35 лучших признаков")
    print("Топ-10:", selected[:10])

# Сохраняем список лучших признаков
os.makedirs('results', exist_ok=True)
pd.DataFrame(top_features).to_csv('results/top_features.csv', index=False)

print("\nОтбор завершён!")
print("Лучшие 35 признаков сохранены в results/top_features.csv")
print("\nТеперь мы готовы переходить к первой модели.")