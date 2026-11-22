import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

from sections.signals_section import show_signals_section
from sections.energy_section import show_energy_section
from sections.crypto_section import show_crypto_section
from sections.co2_section import show_co2_section
from sections.etf_section import show_etf_section   # ← Новый блок

st_autorefresh(interval=45000, key="main_refresh")

st.set_page_config(page_title="HorizonForge", layout="wide")
st.title("🌍 HorizonForge — Global Intelligence System")
st.caption(f"Обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

show_signals_section()
show_energy_section()
show_crypto_section()
show_co2_section()
show_etf_section()          # ← Добавлен

st.caption("HorizonForge © 2026")