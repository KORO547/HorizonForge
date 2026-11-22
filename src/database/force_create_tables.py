from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql+psycopg2://horizonforge:horizon123@localhost:5432/horizonforge"
)

print("🔄 Принудительное создание таблиц...")

with engine.connect() as conn:
    # Создаём assets
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS assets (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50) NOT NULL,
            ticker VARCHAR(20) NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    
    conn.execute(text("""
        INSERT INTO assets (id, name, ticker) 
        VALUES (1, 'Bitcoin', 'BTC'), (2, 'Ethereum', 'ETH'), (3, 'Solana', 'SOL')
        ON CONFLICT (id) DO NOTHING;
    """))
    
    # Создаём market_prices
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS market_prices (
            time TIMESTAMP NOT NULL,
            asset_id INTEGER NOT NULL,
            price NUMERIC NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            volume BIGINT NOT NULL,
            PRIMARY KEY (time, asset_id)
        );
    """))
    
    print("✅ Обе таблицы успешно созданы!")

print("Готово!")