import pandas as pd
import os

print("=== Финальный отбор топ-признаков ===\n")

top_n = 30  # сколько лучших признаков оставляем

for symbol_file in ['BTC_USDT_1h_features.csv', 'ETH_USDT_1h_features.csv']:
    file_path = f'data/features/{symbol_file}'
    df = pd.read_csv(file_path)
    
    feature_cols = [col for col in df.columns 
                    if col not in ['timestamp', 'target', 'target_dir']]
    
    symbol_name = symbol_file.replace('_1h_features.csv', '').replace('_', '/')
    
    # Корреляция
    corr = df[feature_cols + ['target']].corr()['target'].abs()
    corr_sorted = corr.sort_values(ascending=False)
    
    # Берём топ-N признаков
    selected_features = corr_sorted.head(top_n + 1).index.tolist()
    selected_features = [col for col in selected_features if col != 'target']
    
    print(f"\n{symbol_name} — отобрано {len(selected_features)} лучших признаков")
    print("Топ-10:", selected_features[:10])
    
    # Сохраняем список лучших признаков
    os.makedirs('results', exist_ok=True)
    pd.Series(selected_features, name='feature').to_csv(f'results/top_features_{symbol_name.replace("/", "_")}.csv', index=False)

print("\nФинальный отбор завершён!")
print(f"Используем топ-{top_n} признаков для каждой монеты.")
print("Теперь мы готовы к обучению модели на очищенном наборе.")