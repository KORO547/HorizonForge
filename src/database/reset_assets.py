from sqlalchemy import create_engine, text

engine = create_engine('sqlite:///horizonforge.db', echo=True)

def reset_assets():
    print("🔄 Полностью пересоздаём таблицу assets...")
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS assets"))
        conn.execute(text("""
            CREATE TABLE assets (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                ticker TEXT NOT NULL UNIQUE
            )
        """))
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
        conn.commit()   # ← Это важно!
        print("✅ Добавлено 7 монет!")

if __name__ == "__main__":
    reset_assets()