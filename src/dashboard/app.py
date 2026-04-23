import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="HorizonForge Dashboard", layout="wide")
st.title("HorizonForge - Global Market Forecast")

st.write(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Загружаем последние предсказания
try:
    btc = pd.read_csv('data/features/BTC_USDT_features_v6.2.csv')
    eth = pd.read_csv('data/features/ETH_USDT_features_v6.2.csv')
    
    st.subheader("Current Signals (6h horizon)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**BTC/USDT**")
        latest_btc = btc.iloc[-1]
        prob = latest_btc.get('prob_up', 'N/A')  # если есть
        st.metric("Direction 6h", "UP" if latest_btc['target'] == 1 else "DOWN", 
                  f"Imbalance: {latest_btc.get('ob_imbalance', 0):.3f}")
    
    with col2:
        st.write("**ETH/USDT**")
        latest_eth = eth.iloc[-1]
        st.metric("Direction 6h", "UP" if latest_eth['target'] == 1 else "DOWN", 
                  f"Imbalance: {latest_eth.get('ob_imbalance', 0):.3f}")

except:
    st.warning("No prediction data yet. Run automation first.")

st.write("---")
st.caption("HorizonForge Automation is running every hour.")