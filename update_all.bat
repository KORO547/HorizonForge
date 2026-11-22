@echo off
chcp 65001 >nul
cd /d C:\Projects\HorizonForge\HorizonForge

echo ========================================
echo     HorizonForge - Полное обновление
echo ========================================
echo.

echo [1] Скачивание данных + Order Book...
python src/markets/crypto/download_data_with_ob.py

echo [2] Генерация признаков...
python src/features/engineer_features.py

echo [3] Обновление энергетических данных...
python src/macro/energy_data.py

echo [4] Обновление крипто-факторов...
python src/crypto/crypto_features.py

echo [5] Обучение полной модели крипты...
python src/crypto/crypto_predictor_full.py

echo [6] Создание текущих предсказаний...
python src/prediction/current_predictions.py

echo.
echo ========================================
echo     Обновление успешно завершено!
echo     %date% %time%
echo ========================================
pause