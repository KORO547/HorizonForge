import pandas as pd

print("=== Анализ текущей климатической ситуации (на 2026-04-22) ===\n")

# Загружаем климатические данные
climate = pd.read_csv('data/macro/climate/climate_daily.csv')
climate['timestamp'] = pd.to_datetime(climate['timestamp'])

# Берём последнюю дату
latest = climate.iloc[-1].copy()
latest_date = latest['timestamp'].date()

print(f"Последняя доступная дата: {latest_date}\n")

print("ТЕКУЩИЕ КЛИМАТИЧЕСКИЕ ПОКАЗАТЕЛИ:")
print("=" * 55)
print(f"Глобальная температурная аномалия     : {latest['global_temp_anomaly']:.3f} °C")
print(f"Температурная аномалия по США        : {latest['us_temp_anomaly']:.3f} °C")
print(f"Аномалия осадков                     : {latest['precipitation_anomaly']:.3f}")
print(f"Индекс засухи                        : {latest['drought_index']:.2f} / 7.0")
print(f"Дней с экстремальной жарой           : {int(latest['extreme_heat_days'])}")
print(f"Дней с экстремальными дождями        : {int(latest['extreme_rain_days'])}")

# Оценка ситуации
drought = latest['drought_index']
temp_us = latest['us_temp_anomaly']
precip = latest['precipitation_anomaly']

print("\nОЦЕНКА ТЕКУЩЕГО КЛИМАТИЧЕСКОГО ФОНА:")
print("-" * 55)

if drought > 4.5:
    print("⚠️  СИЛЬНАЯ ЗАСуха — высокий риск для сельского хозяйства")
elif drought > 3.0:
    print("⚠️  Умеренная засуха — осторожно для агросектора")
else:
    print("✅ Засуха в норме или ниже нормы")

if temp_us > 1.8:
    print("⚠️  Высокая температура в США — риск для урожая и энергопотребления")
elif temp_us > 1.2:
    print("⚠️  Повышенная температура")
else:
    print("✅ Температура в пределах нормы")

if precip < -1.2:
    print("⚠️  Значительный дефицит осадков")
elif precip > 1.2:
    print("✅ Избыток осадков (риск наводнений)")
else:
    print("✅ Осадки в нормальном диапазоне")

# Общий вывод
risk_level = "НЕГАТИВНЫЙ" if (drought > 4.0 or temp_us > 2.0 or precip < -1.5) else \
             "УМЕРЕННО НЕГАТИВНЫЙ" if (drought > 3.0 or temp_us > 1.5) else "НЕЙТРАЛЬНЫЙ"

print(f"\n→ Общий климатический фон: **{risk_level}**")

# Сохраняем
latest.to_frame().T.to_csv('results/current_climate_situation.csv', index=False)
print("\nТекущая климатическая ситуация сохранена в: results/current_climate_situation.csv")