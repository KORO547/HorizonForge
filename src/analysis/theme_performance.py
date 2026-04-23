import pandas as pd
import numpy as np
import joblib

print("=== Анализ эффективности модели по темам ===\n")

# Загружаем данные и модель
df = pd.read_csv('data/features/stocks/global_panel_with_deps.csv')
model = joblib.load('models/global_dependency_model_v2.pkl')

feature_cols = [
    'rsi_14', 'macd', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower',
    'atr_14', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'price_range', 'vol_10', 'volume_ratio',
    'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_10', 'close_lag_20',
    'market_return', 'sector_return',
    'relative_to_market', 'relative_to_sector',
    'market_return_lag1', 'sector_return_lag1',
    'dayofweek'
]

X = df[feature_cols]
df['prob'] = model.predict_proba(X)[:, 1]
df['prediction'] = (df['prob'] > 0.60).astype(int)   # порог 0.60 для анализа

# Определяем темы (группы)
theme_map = {
    # Сельское хозяйство и удобрения
    'CTVA':'Сельхоз', 'MOS':'Удобрения', 'CF':'Удобрения', 'NTR':'Удобрения', 
    'ADM':'Агробизнес', 'BG':'Агробизнес', 'FMC':'Агрохимия', 'AGCO':'Сельхозтехника',
    
    # Вода и опреснение
    'XYL':'Вода', 'PNR':'Вода', 'AWK':'Вода', 'WTRG':'Вода', 'AOS':'Вода',
    
    # Экология / Зеленая энергия
    'ENPH':'Зеленая энергия', 'FSLR':'Зеленая энергия', 'BEPC':'Зеленая энергия',
    'PLUG':'Водород', 'BLDP':'Водород', 'SEDG':'Зеленая энергия', 'RUN':'Зеленая энергия',
    
    # Телеком
    'TMUS':'Телеком', 'CMCSA':'Телеком', 'CHTR':'Телеком', 'LUMN':'Телеком',
    
    # Энергетика и топливо
    'XOM':'Энергетика', 'CVX':'Энергетика', 'COP':'Энергетика', 'SLB':'Энергетика',
    
    # Финансы
    'V':'Финансы', 'MA':'Финансы', 'JPM':'Финансы',
    
    # Остальное
    'default':'Другие'
}

df['theme'] = df['ticker'].map(theme_map).fillna('Другие')

# Анализ по темам
theme_stats = df.groupby('theme').agg({
    'prob': ['mean', 'std', 'count'],
    'target': 'mean'
}).round(4)

theme_stats.columns = ['Средняя вероятность', 'Std вероятности', 'Кол-во наблюдений', 'Средняя реализация (target)']
theme_stats = theme_stats.sort_values('Средняя вероятность', ascending=False)

print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ ПО ТЕМАМ (средняя вероятность роста)")
print("="*80)
print(theme_stats)

# Топ-акции по вероятности внутри тем
print("\n\nТоп-5 акций с самой высокой средней вероятностью:")
top_stocks = df.groupby('ticker').agg({
    'prob': 'mean',
    'theme': 'first'
}).sort_values('prob', ascending=False).head(10)

print(top_stocks)

# Сохраняем результаты
theme_stats.to_csv('results/theme_performance.csv')
print("\nРезультаты анализа по темам сохранены в: results/theme_performance.csv")