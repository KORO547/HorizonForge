import streamlit as st

def show_co2_section():
    st.header("🌍 Климат и CO₂")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Глобальная температура", "+0.74°C", "аномалия")
    with col2:
        st.metric("CO₂ выбросы", "37,800 Mt", "2025")
    
    st.warning("⚠️ Высокая температура и засуха могут повлиять на энергетику и майнинг")