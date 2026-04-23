import pandas as pd
import os

print("=== Анализ всех 59 признаков (v2) ===\n")

results = {}

for symbol_file in ['BTC_USDT_1h_features.csv', 'ETH_USDT_1h_features.csv']:
    file_path = f'data/features/{symbol_file}'
    df = pd.read_csv(file_path)
    
    feature_cols = [col for col in df.columns 
                    if col not in ['timestamp', 'target', 'target_dir']]
    
    symbol_name = symbol_file.replace('_1h_features.csv', '').replace('_', '/')
    
    print(f"\n{symbol_name} — {len(feature_cols)} признаков")
    print("-" * 70)
    
    # Корреляция с target
    corr = df[feature_cols + ['target']].corr()['target'].abs()
    corr_sorted = corr.sort_values(ascending=False)
    
    print("Топ-20 самых сильных признаков:")
    print(corr_sorted.head(20))
    
    # Сохраняем топ-признаки
    os.makedirs('results', exist_ok=True)
    corr_sorted.to_csv(f'results/{symbol_name.replace("/", "_")}_full_corr.csv')
    
    # Сохраняем только топ-30 признаков для дальнейшей работы
    top_30 = corr_sorted.head(30).index.tolist()
    top_30 = [col for col in top_30 if col != 'target']
    
    results[symbol_name] = top_30

print("\n" + "="*70)
print("Анализ завершён!")
print("Топ-признаки сохранены в папке 'results/'")
print("\nРекомендация: дальше использовать только топ-25–35 признаков.")

# Выводим общие топ-признаки для обоих символов
print("\nОбщие сильные признаки (пересечение для BTC и ETH):")
common = set(results['BTC/USDT'][:30]) & set(results['ETH/USDT'][:30])
print(list(common)[:15])