import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Пути к файлам
btc_file = 'data/raw/BTC_USDT_1h.csv'
eth_file = 'data/raw/ETH_USDT_1h.csv'

# Загружаем данные
print("Загружаем данные...")
btc = pd.read_csv(btc_file)
eth = pd.read_csv(eth_file)

# Преобразуем timestamp в datetime
btc['timestamp'] = pd.to_datetime(btc['timestamp'])
eth['timestamp'] = pd.to_datetime(eth['timestamp'])

print(f"\nBTC/USDT: {len(btc)} свечей")
print(f"Период: с {btc['timestamp'].iloc[0]} по {btc['timestamp'].iloc[-1]}")
print(f"ETH/USDT: {len(eth)} свечей")
print(f"Период: с {eth['timestamp'].iloc[0]} по {eth['timestamp'].iloc[-1]}\n")

# Строим график
fig = make_subplots(rows=2, cols=1, 
                    subplot_titles=('BTC/USDT', 'ETH/USDT'),
                    vertical_spacing=0.12)

# BTC
fig.add_trace(go.Candlestick(x=btc['timestamp'],
                             open=btc['open'],
                             high=btc['high'],
                             low=btc['low'],
                             close=btc['close'],
                             name='BTC/USDT'),
              row=1, col=1)

# ETH
fig.add_trace(go.Candlestick(x=eth['timestamp'],
                             open=eth['open'],
                             high=eth['high'],
                             low=eth['low'],
                             close=eth['close'],
                             name='ETH/USDT'),
              row=2, col=1)

fig.update_layout(
    title='HorizonForge — Исторические данные BTC и ETH (1h)',
    height=800,
    template='plotly_dark',
    showlegend=True
)

fig.update_xaxes(rangeslider_visible=False)

print("Открываем график...")
fig.show()