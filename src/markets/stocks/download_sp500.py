import yfinance as yf
import os
from tqdm import tqdm

print("=== Расширение universe по темам: Сельхоз + Вода + Экология + Телеком ===\n")

# Новый список компаний для добавления
new_tickers = [
    # 1. Сельское хозяйство и удобрения
    'CTVA', 'MOS', 'CF', 'NTR', 'ADM', 'BG', 'FMC', 'AGCO', 'CNHI',
    
    # 2. Вода и опреснение
    'XYL', 'PNR', 'AWK', 'WTRG', 'AOS', 'BMI', 'CWCO',
    
    # 3. Экологические / Зеленые компании
    'ENPH', 'FSLR', 'BEPC', 'PLUG', 'BLDP', 'SEDG', 'RUN', 'CWEN',
    
    # 4. Телекоммуникации
    'TMUS', 'CMCSA', 'CHTR', 'DISH', 'LUMN', 'RCI', 'TU', 'BCE', 'AMX'
]

print(f"Добавляем {len(new_tickers)} новых компаний по темам")

os.makedirs('data/raw/stocks/sp500', exist_ok=True)

successful = 0
already_exists = 0
failed = []

for ticker in tqdm(new_tickers):
    filepath = f'data/raw/stocks/sp500/{ticker}_1d.csv'
    
    # Проверяем, не скачана ли уже компания
    if os.path.exists(filepath):
        already_exists += 1
        continue
    
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        
        if df.empty:
            failed.append(ticker)
            continue
        
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'timestamp'}, inplace=True)
        
        df.to_csv(filepath, index=False)
        successful += 1
        
    except Exception as e:
        failed.append(ticker)
        print(f"Ошибка при скачивании {ticker}: {e}")

print("\nДобавление тематических компаний завершено!")
print(f"Успешно скачано новых: {successful}")
print(f"Уже было скачано: {already_exists}")
print(f"Не удалось скачать: {len(failed)}")

if failed:
    print("Неудачные тикеры:", failed)

print(f"\nОбщее количество компаний в папке sp500: {len(os.listdir('data/raw/stocks/sp500'))}")