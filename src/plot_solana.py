import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta

# Загружаем данные
df = pd.read_csv('data/raw/SOL_USDT_1h.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# === Индикаторы ===
df['ema_9'] = ta.ema(df['close'], length=9)
df['ema_21'] = ta.ema(df['close'], length=21)
df['rsi'] = ta.rsi(df['close'], length=14)

# Bollinger Bands — безопасный способ
bb = ta.bbands(df['close'], length=20, std=2)
df['bb_lower'] = bb.iloc[:, 0]   # нижняя полоса
df['bb_middle'] = bb.iloc[:, 1]  # средняя
df['bb_upper'] = bb.iloc[:, 2]   # верхняя

print("Колонки Bollinger Bands:", bb.columns.tolist())  # для проверки

# === График ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1], sharex=True)

ax1.plot(df.index, df['close'], label='SOL/USDT Close', color='blue', linewidth=2)
ax1.plot(df.index, df['ema_9'], label='EMA 9', color='orange', linewidth=1.5)
ax1.plot(df.index, df['ema_21'], label='EMA 21', color='red', linewidth=1.5)
ax1.plot(df.index, df['bb_upper'], color='gray', linestyle='--', alpha=0.7)
ax1.plot(df.index, df['bb_lower'], color='gray', linestyle='--', alpha=0.7)
ax1.fill_between(df.index, df['bb_lower'], df['bb_upper'], color='gray', alpha=0.12)

ax1.set_title('SOL/USDT — График с EMA и Bollinger Bands', fontsize=16, pad=20)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# RSI
ax2.plot(df.index, df['rsi'], label='RSI 14', color='purple')
ax2.axhline(70, color='red', linestyle='--', alpha=0.7)
ax2.axhline(30, color='green', linestyle='--', alpha=0.7)
ax2.set_ylim(0, 100)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Сохраняем
plt.savefig('results/solana_chart.png', dpi=300, bbox_inches='tight')
print(f"\n✅ График успешно сохранён: results/solana_chart.png")
print(f"Период: {df.index[0]} — {df.index[-1]}")
print(f"Всего свечей: {len(df)}")