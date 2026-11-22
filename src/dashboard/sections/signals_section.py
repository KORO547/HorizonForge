import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine("postgresql+psycopg2://horizonforge:horizon123@localhost:5432/horizonforge")

def show_signals_section():
    st.header("📈 Рыночные Сигналы")
    
    query = """
        SELECT time, asset_id, price 
        FROM market_prices 
        ORDER BY time DESC LIMIT 10
    """
    df = pd.read_sql(query, engine)
    
    asset_names = {1: "Bitcoin", 2: "Ethereum", 3: "Solana"}
    df['asset'] = df['asset_id'].map(asset_names)
    
    st.dataframe(df[['time', 'asset', 'price']], use_container_width=True)
    
    st.info("🔄 Сигналы обновляются каждые 45 секунд")