-- HorizonForge Database Schema v1.0

CREATE TABLE IF NOT EXISTS assets (
    asset_id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100),
    asset_type VARCHAR(20) CHECK (asset_type IN ('crypto', 'stock', 'commodity', 'macro')),
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS market_prices (
    time TIMESTAMPTZ NOT NULL,
    asset_id INTEGER REFERENCES assets(asset_id),
    price DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,
    open_price DOUBLE PRECISION,
    high_price DOUBLE PRECISION,
    low_price DOUBLE PRECISION,
    PRIMARY KEY (time, asset_id)
);

CREATE TABLE IF NOT EXISTS macro_indicators (
    time TIMESTAMPTZ NOT NULL,
    indicator_name VARCHAR(100),
    country VARCHAR(50),
    value DOUBLE PRECISION,
    source VARCHAR(50),
    PRIMARY KEY (time, indicator_name, country)
);

CREATE TABLE IF NOT EXISTS geopolitical_events (
    event_id SERIAL PRIMARY KEY,
    event_date DATE NOT NULL,
    title TEXT,
    description TEXT,
    impact_level INTEGER CHECK (impact_level BETWEEN 1 AND 5),
    affected_tickers TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Индексы для скорости
CREATE INDEX idx_market_prices_time ON market_prices(time DESC);
CREATE INDEX idx_macro_time ON macro_indicators(time DESC);