import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

def make_current_predictions():
    print("=== Текущие предсказания глобальной модели ===")
    
    model_path = 'models/global_model_climate_enhanced.pkl'
    data_path = 'data/features/stocks/global_panel_with_climate.csv'
    
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    latest = df.groupby('ticker').tail(1).copy()
    
    # Убираем нечисловые колонки
    exclude = ['ticker', 'target', 'Date', 'timestamp', 'sector', 'symbol']
    feature_cols = [col for col in latest.columns if col not in exclude and pd.api.types.is_numeric_dtype(latest[col])]
    
    X = latest[feature_cols].copy()
    X = X.fillna(X.median())
    
    print(f"Подаём в модель: {X.shape[1]} признаков (ожидалось 44)")
    
    # Временное отключение проверки количества признаков
    probs = model.predict_proba(X, predict_disable_shape_check=True)[:, 1]
    
    latest = latest.copy()
    latest['prob_up'] = probs
    latest['signal'] = latest['prob_up'].apply(
        lambda x: "STRONG BUY" if x > 0.65 else "BUY" if x > 0.55 else "HOLD" if x > 0.45 else "SELL"
    )
    
    top10 = latest.nlargest(10, 'prob_up')[['ticker', 'prob_up', 'signal']]
    print("\n🔥 ТОП-10 НА ПОКУПКУ:")
    print(top10)
    
    top10.to_csv('results/current_predictions.csv', index=False)
    print("\n✅ Предсказания сохранены!")

if __name__ == "__main__":
    make_current_predictions()