from sqlalchemy import create_engine, text
from datetime import datetime

engine = create_engine("sqlite:///horizonforge.db")

def load_crypto_data():
    print(">>> Starting data update (SQLite)...")

    assets = {1: "Bitcoin", 2: "Ethereum", 3: "Solana", 4: "Dogecoin", 5: "Ripple", 6: "Cardano", 7: "Binance Coin"}
    
    for asset_id, name in assets.items():
        print(f"-> Updating {name}...")
        try:
            # Пока имитируем обновление
            print(f"   [OK] {name} - data updated")
        except Exception as e:
            print(f"   [ERROR] {name}: {e}")

    print(">>> Data update completed.\n")

if __name__ == "__main__":
    load_crypto_data()