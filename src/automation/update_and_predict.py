import subprocess
import time
from datetime import datetime

print("=== HorizonForge Automation Started ===")
print(f"Start time: {datetime.now()}\n")

def run_command(cmd, name):
    print(f"-> Running {name}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode == 0:
            print(f"SUCCESS: {name}")
        else:
            print(f"ERROR: {name}")
            print(result.stderr[-800:])   # last 800 characters of error
    except Exception as e:
        print(f"ERROR running {name}: {e}")

while True:
    print(f"\n=== New Cycle - {datetime.now()} ===\n")
    
    run_command("python src/markets/crypto/download_data_with_ob.py", "Download Data + OB")
    run_command("python src/features/engineer_features.py", "Feature Engineering")
    run_command("python src/models/model_6h_v3.py", "Model Prediction")
    
    print(f"\nCycle finished. Next update in 1 hour...\n")
    time.sleep(3600)