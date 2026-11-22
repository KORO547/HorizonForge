from sqlalchemy import create_engine, text
import os

# SQLite база в папке проекта
db_path = "horizonforge.db"
engine = create_engine(f"sqlite:///{db_path}")

def init_database():
    print("🗄️ Инициализация SQLite базы...")

    with engine.connect() as conn:
        # Таблица assets
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS assets (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                ticker TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        # Добавляем монеты
        conn.execute(text("""
            INSERT OR IGNORE INTO assets (id, name, ticker) VALUES 
            (1, 'Bitcoin', 'BTC'),
            (2, 'Ethereum', 'ETH'),
            (3, 'Solana', 'SOL'),
            (4, 'Dogecoin', 'DOGE'),
            (5, 'Ripple', 'XRP'),
            (6, 'Cardano', 'ADA'),
            (7, 'Binance Coin', 'BNB');
        """))

        print(f"✅ База SQLite создана: {db_path}")
        print("   Добавлено 7 активов")

if __name__ == "__main__":
    init_database()