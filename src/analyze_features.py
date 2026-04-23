import pandas as pd
import os

print("=== Улучшенный анализ качества признаков ===\n")

for symbol in ['BTC_USDT', 'ETH_USDT']:
    file_path = f'data/features/{symbol}_1h_features.csv'
    df = pd.read_csv(file_path)
    
    # Все признаки кроме target
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'target', 'target_dir']]
    
    print(f"\n{symbol.replace('_', '/')} — {len(feature_cols)} признаков")
    print("-" * 60)
    
    # Корреляция с будущей доходностью
    corr = df[feature_cols + ['target']].corr()['target'].abs()
    corr = corr.sort_values(ascending=False)
    
    print("Топ-15 самых сильных признаков:")
    print(corr.head(15))
    
    # Сохраняем результат
    os.makedirs('results', exist_ok=True)
    corr.to_csv(f'results/{symbol}_feature_corr.csv')

print("\nАнализ завершён!")
print("Результаты сохранены в папке 'results/'")