import streamlit as st
import pandas as pd
from datetime import datetime
import joblib

st.set_page_config(page_title="HorizonForge", layout="wide", page_icon="🌍")
st.title("🌍 HorizonForge - Global Market Forecast")
st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Automation: every hour")

# ====================== CURRENT SIGNALS ======================
st.subheader("📈 Current Signals (6h horizon)")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("BTC/USDT", "DOWN", "Imbalance: 0.000")
with col2:
    st.metric("ETH/USDT", "DOWN", "Imbalance: 0.000")
with col3:
    st.metric("SOL/USDT", "NEUTRAL", "Imbalance: -0.097")

# ====================== TOP SIGNALS FROM GLOBAL MODEL ======================
st.subheader("🔥 Top Signals Today (Global Model with Climate)")

# Здесь пока заглушка, потом заменим на реальные данные из модели
top_signals = pd.DataFrame({
    "Ticker": ["DG", "V", "ASML", "DHR", "TTWO", "ENPH", "XYL"],
    "Probability": [0.68, 0.62, 0.61, 0.59, 0.58, 0.57, 0.56],
    "Signal": ["Strong Buy", "Buy", "Buy", "Buy", "Buy", "Buy", "Buy"],
    "Theme": ["Retail", "Finance", "Semiconductors", "Healthcare", "Gaming", "Green Energy", "Water"]
})

st.dataframe(top_signals, use_container_width=True, hide_index=True)

# ====================== CLIMATE SITUATION ======================
st.subheader("🌡️ Current Climate Situation")

col4, col5, col6, col7 = st.columns(4)

with col4:
    st.metric("Global Temp Anomaly", "+0.75°C", "Moderately elevated")
with col5:
    st.metric("US Temp Anomaly", "+1.75°C", "Elevated")
with col6:
    st.metric("Drought Index", "0.65 / 7.0", "Low")
with col7:
    st.metric("Precipitation", "+2.26", "Excess (flood risk)")

st.info("**Overall Climate Background: Moderately Negative** — Elevated temperature + excess precipitation may pressure agriculture and increase energy demand.")

# ====================== MACRO & RISK OVERVIEW ======================
st.subheader("⚠️ Macro & Risk Overview")

col8, col9 = st.columns(2)

with col8:
    st.write("**Key Risks**")
    st.write("- Elevated US temperature → higher energy consumption")
    st.write("- Excess rainfall → potential flooding in agriculture regions")
    st.write("- Geopolitical tensions (ongoing)")

with col9:
    st.write("**Opportunities**")
    st.write("- Green energy & water technology companies")
    st.write("- Defensive sectors (utilities, healthcare)")
    st.write("- Companies resilient to extreme weather")

# ====================== FOOTER ======================
st.caption("HorizonForge — Global Multi-Market Alpha Prediction System | Combining Technical Analysis, Market Dependencies & Climate Factors")

with st.sidebar:
    st.header("About HorizonForge")
    st.write("186 stocks + Crypto + Climate-aware AI model")
    st.write("Current Model AUC: **0.738**")
    st.write("Last Walk-Forward Return: **+112%** (drawdown -8.8%)")
    
    st.markdown("---")
    st.write("**Themes monitored:**")
    st.write("• Agriculture & Fertilizers")
    st.write("• Water & Desalination")
    st.write("• Green Energy & Ecology")
    st.write("• Telecom")
    st.write("• Energy & Oil")
    
    if st.button("🔄 Refresh All Data"):
        st.rerun()