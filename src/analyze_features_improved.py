import pandas as pd
import os

print("=== Улучшенный анализ качества признаков ===\n")

for symbol_file in ['BTC_USDT_1h_features.csv', 'ETH_USDT_1h_features.csv']:
    file_path = f'data/features/{symbol_file}'
    
    if not os.path.exists(file_path):
        print(f"Файл {symbol_file} не найден!")
        continue
        
    df = pd.read_csv(file_path)
    
    feature_cols = [col for col in df.columns 
                    if col not in ['timestamp', 'target', 'target_dir']]
    
    symbol_name = symbol_file.replace('_1h_features.csv', '').replace('_', '/')
    
    print(f"\n{symbol_name} — {len(feature_cols)} признаков")
    print("-" * 60)
    
    # Корреляция признаков с target
    corr = df[feature_cols + ['target']].corr()['target'].abs()
    corr_sorted = corr.sort_values(ascending=False)
    
    print("Топ-15 самых сильных признаков:")
    print(corr_sorted.head(15))
    
    # Сохраняем результат
    os.makedirs('results', exist_ok=True)
    corr_sorted.to_csv(f'results/{symbol_name.replace("/", "_")}_corr.csv')

print("\nАнализ завершён!")
print("Результаты сохранены в папке 'results/'")