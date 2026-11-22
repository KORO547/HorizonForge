import time
import subprocess
from datetime import datetime

print("🔄 Автообновление HorizonForge запущено")
print("Обновление данных каждые 30 минут\n")

count = 0

try:
    while True:
        count += 1
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] #{count} — Обновление данных...")

        try:
            result = subprocess.run(
                ["python", "load_data.py"],
                cwd="src/database",
                capture_output=True,
                text=True,
                timeout=90
            )
            print(result.stdout.strip())
            if result.stderr:
                print("WARNING:", result.stderr.strip())
        except Exception as e:
            print(f"Ошибка: {e}")

        print("-" * 60)
        time.sleep(1800)  # 30 минут

except KeyboardInterrupt:
    print("\n\n⛔ Автообновление остановлено вручную.")
except Exception as e:
    print(f"Критическая ошибка: {e}")