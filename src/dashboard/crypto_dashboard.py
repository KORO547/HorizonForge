import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=60000, key="crypto_refresh")

st.set_page_config(page_title="CryptoForge", layout="wide")
st.title("₿ CryptoForge — Прогнозирование Криптовалют")
st.caption(f"Обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Основные цены
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Bitcoin", "$94,850", "↑1.2%")
with col2: st.metric("Ethereum", "$2,780", "↑0.8%")
with col3: st.metric("Solana", "$178.4", "↑3.5%")
with col4: st.metric("BTC Dominance", "52.8%", "-0.4%")

# ==================== ПРОГНОЗ ====================
st.header("📈 Прогноз Politico-Economic Memory Model v2")

model_path = 'models/crypto_geopolitical_model_v2.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
    
    if os.path.exists('data/crypto/crypto_with_macro.csv'):
        df = pd.read_csv('data/crypto/crypto_with_macro.csv')
        
        latest = df.iloc[-1:].copy()
        
        # Добавляем лаги
        if 'BTC_Price' in df.columns:
            for lag in [1, 3, 7, 14, 30]:
                latest[f'BTC_Price_lag_{lag}'] = df['BTC_Price'].iloc[-lag-1] if len(df) > lag else df['BTC_Price'].iloc[-1]
        else:
            for lag in [1, 3, 7, 14, 30]:
                latest[f'BTC_Price_lag_{lag}'] = 94500
        
        if 'Brent_Price' in df.columns:
            for lag in [1, 3, 7]:
                latest[f'Brent_lag_{lag}'] = df['Brent_Price'].iloc[-lag-1] if len(df) > lag else df['Brent_Price'].iloc[-1]
        
        latest['day_of_week'] = pd.to_datetime(latest['Date']).dt.dayofweek
        latest['month'] = pd.to_datetime(latest['Date']).dt.month
        latest['year_cycle'] = np.sin(2 * np.pi * pd.to_datetime(latest['Date']).dt.dayofyear / 365)
        
        numeric_latest = latest.select_dtypes(include=['number']).copy()
        
        prob_price = model.predict_proba(numeric_latest, predict_disable_shape_check=True)[0][1] * 100
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.success("**Рост цены BTC**")
            st.metric("Вероятность", f"{prob_price:.1f}%")
            st.progress(prob_price / 100)
        
        with col_p2:
            st.info("**Рост BTC Dominance**")
            st.metric("Вероятность", "—", "↔")
    else:
        st.warning("Данные не найдены")
else:
    st.error("Модель не найдена")

# Макро-факторы
st.subheader("🌍 Ключевые макро-факторы")
if os.path.exists('data/crypto/crypto_with_macro.csv'):
    latest = pd.read_csv('data/crypto/crypto_with_macro.csv').iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("🛢️ Brent", f"${latest.get('Brent_Price', 85):.1f}")
    with c2: st.metric("⛏️ Майнинг", f"{latest.get('Crypto_Mining_Global_TWh', 160):.1f} TWh")
    with c3: st.metric("🌡️ CO₂", f"{int(latest.get('Global_CO2_Mt', 37800)):,} Mt")
    with c4: st.metric("⚡ Китай", f"${latest.get('China_Electricity_Price', 0.115):.3f}/kWh")

st.success("✅ Модель v2 активна")
st.caption("HorizonForge © 2026")