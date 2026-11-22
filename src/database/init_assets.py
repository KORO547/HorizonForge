from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql+psycopg2://horizonforge:horizon123@localhost:5432/horizonforge"
)

def init_assets():
    print("📋 Расширяем список активов...")

    with engine.connect() as conn:
        # Создаём/обновляем таблицу
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS assets (
                id SERIAL PRIMARY KEY,
                name VARCHAR(50) NOT NULL,
                ticker VARCHAR(20) NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        # Расширенный список
        conn.execute(text("""
            INSERT INTO assets (id, name, ticker) VALUES 
                (1, 'Bitcoin', 'BTC'),
                (2, 'Ethereum', 'ETH'),
                (3, 'Solana', 'SOL'),
                (4, 'Dogecoin', 'DOGE'),
                (5, 'Ripple', 'XRP'),
                (6, 'Cardano', 'ADA'),
                (7, 'Binance Coin', 'BNB')
            ON CONFLICT (id) DO NOTHING;
        """))

        print("✅ Добавлено 7 активов")

if __name__ == "__main__":
    init_assets()