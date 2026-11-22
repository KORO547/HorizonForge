from sqlalchemy import create_engine, text

engine = create_engine('sqlite:///horizonforge.db')

def reset_database():
    print("🗑️ Полный сброс базы данных...")

    with engine.connect() as conn:
        # Удаляем старые таблицы
        conn.execute(text("DROP TABLE IF EXISTS market_prices"))
        conn.execute(text("DROP TABLE IF EXISTS assets"))
        
        # Создаём таблицу активов
        conn.execute(text("""
            CREATE TABLE assets (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                ticker TEXT NOT NULL UNIQUE
            )
        """))
        
        # Создаём таблицу цен
        conn.execute(text("""
            CREATE TABLE market_prices (
                time TEXT NOT NULL,
                asset_id INTEGER NOT NULL,
                price REAL NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                volume REAL,
                PRIMARY KEY (time, asset_id)
            )
        """))
        
        # Добавляем 7 монет
        conn.execute(text("""
            INSERT INTO assets (id, name, ticker) VALUES
            (1, 'Bitcoin', 'BTC'),
            (2, 'Ethereum', 'ETH'),
            (3, 'Solana', 'SOL'),
            (4, 'Dogecoin', 'DOGE'),
            (5, 'Ripple', 'XRP'),
            (6, 'Cardano', 'ADA'),
            (7, 'Binance Coin', 'BNB')
        """))
        
        conn.commit()
    
    print("✅ База данных полностью сброшена и готова!")

if __name__ == "__main__":
    reset_database()