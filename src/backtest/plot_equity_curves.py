import pandas as pd
import matplotlib.pyplot as plt
import os

print("=== Построение графиков Equity Curve для BTC, ETH, SOL ===\n")

pairs = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']
plt.figure(figsize=(14, 8))

for pair in pairs:
    try:
        # Пробуем загрузить данные с предсказаниями (если есть)
        df = pd.read_csv(f'data/features/{pair}_features_v6.3.csv')
        
        # Простая кумулятивная доходность на основе target (для сравнения)
        df['return'] = df['close'].pct_change()
        df['strategy'] = df['target'].shift(1) * df['return']   # сдвиг, чтобы не смотреть в будущее
        
        df['cum_return'] = (1 + df['strategy']).cumprod()
        df['equity'] = 10000 * df['cum_return']
        
        plt.plot(df['timestamp'], df['equity'], label=f'{pair.replace("_", "/")} (6h model)')
        
        final_equity = df['equity'].iloc[-1]
        print(f"{pair.replace('_', '/')} → Final capital: ${final_equity:.2f} | Return: {(final_equity/10000-1)*100:.1f}%")
        
    except Exception as e:
        print(f"Ошибка при обработке {pair}: {e}")

plt.title('Equity Curves Comparison - 6h Model (BTC, ETH, SOL)')
plt.xlabel('Дата')
plt.ylabel('Капитал ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nГрафики построены!")