from sqlalchemy import create_engine, text
import time

engine = create_engine(
    "postgresql+psycopg2://horizonforge:horizon123@localhost:5432/horizonforge",
    echo=False
)

print("🔄 Полная принудительная инициализация...")

with engine.connect() as conn:
    conn.execute(text("COMMIT"))  # фиксим предыдущие транзакции
    
    # Удаляем если есть
    conn.execute(text("DROP TABLE IF EXISTS market_prices CASCADE;"))
    conn.execute(text("DROP TABLE IF EXISTS assets CASCADE;"))
    
    # Создаём assets
    conn.execute(text("""
        CREATE TABLE assets (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50) NOT NULL,
            ticker VARCHAR(20) NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    
    # Заполняем assets
    conn.execute(text("""
        INSERT INTO assets (id, name, ticker) VALUES 
        (1, 'Bitcoin', 'BTC'), (2, 'Ethereum', 'ETH'), (3, 'Solana', 'SOL')
        ON CONFLICT DO NOTHING;
    """))
    
    # Создаём market_prices
    conn.execute(text("""
        CREATE TABLE market_prices (
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
    
    conn.execute(text("COMMIT"))
    
    print("✅ Таблицы успешно созданы и зафиксированы!")

# Проверка
with engine.connect() as conn:
    result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"))
    tables = [row[0] for row in result]
    print("Таблицы в базе:", tables)