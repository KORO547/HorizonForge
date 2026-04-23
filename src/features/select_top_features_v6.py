import pandas as pd
import os

print("=== Отбор топ-признаков для v6.0 (6-часовой горизонт) ===\n")

top_n = 25

for symbol_file in ['BTC_USDT_1h_features_v6.csv', 'ETH_USDT_1h_features_v6.csv']:
    file_path = f'data/features/{symbol_file}'
    df = pd.read_csv(file_path)
    
    # Все признаки кроме целевых и времени
    exclude_cols = ['timestamp', 'target', 'target_return_6h']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    symbol_name = symbol_file.replace('_1h_features_v6.csv', '').replace('_', '/')
    
    # Корреляция с новой целью
    corr = df[feature_cols + ['target']].corr()['target'].abs()
    corr_sorted = corr.sort_values(ascending=False)
    
    selected_features = corr_sorted.head(top_n + 1).index.tolist()
    selected_features = [col for col in selected_features if col != 'target']
    
    print(f"\n{symbol_name} — отобрано {len(selected_features)} лучших признаков")
    print("Топ-10:", selected_features[:10])
    
    # Сохраняем
    os.makedirs('results', exist_ok=True)
    pd.Series(selected_features, name='feature').to_csv(
        f'results/top_features_v6_{symbol_name.replace("/", "_")}.csv', 
        index=False
    )

print("\nОтбор признаков для 6-часового горизонта завершён!")
print("Теперь можно переходить к новой модели.")