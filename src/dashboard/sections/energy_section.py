import streamlit as st

def show_energy_section():
    st.header("⚡ Глобальная Энергетика")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Brent Oil", "$84.2", "-0.8%")
    with col2:
        st.metric("Natural Gas (EU)", "€11.4", "+1.2%")
    with col3:
        st.metric("Crypto Mining Power", "162 TWh", "↑")
    
    st.caption("Энергетические факторы сильно влияют на майнинг и цену криптовалют")