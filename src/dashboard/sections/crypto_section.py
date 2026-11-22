import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import plotly.express as px

engine = create_engine("postgresql+psycopg2://horizonforge:horizon123@localhost:5432/horizonforge")

def show_crypto_section():
    st.header("💎 Криптовалюты")
    
    df = pd.read_sql("SELECT * FROM market_prices ORDER BY time DESC LIMIT 60", engine)
    asset_names = {1: "Bitcoin", 2: "Ethereum", 3: "Solana"}
    df['name'] = df['asset_id'].map(asset_names)
    
    for asset_id in [1, 2, 3]:
        asset_df = df[df['asset_id'] == asset_id].head(30).sort_values('time')
        if not asset_df.empty:
            fig = px.line(asset_df, x='time', y='price', title=asset_names[asset_id])
            st.plotly_chart(fig, use_container_width=True)