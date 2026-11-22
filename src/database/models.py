from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, String, Index
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class Asset(Base):
    __tablename__ = 'assets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    ticker = Column(String(20), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    prices = relationship("MarketPrice", back_populates="asset")

    def __repr__(self):
        return f"<Asset {self.ticker}>"


class MarketPrice(Base):
    __tablename__ = 'market_prices'
    
    id = Column(Integer, primary_key=True)
    time = Column(DateTime, nullable=False)
    asset_id = Column(Integer, ForeignKey('assets.id'), nullable=False)
    price = Column(Float, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    volume = Column(Float)
    
    asset = relationship("Asset", back_populates="prices")

    # Уникальный индекс для предотвращения дубликатов
    __table_args__ = (
        Index('ix_market_prices_time_asset', 'time', 'asset_id', unique=True),
    )

    def __repr__(self):
        return f"<Price {self.asset.ticker if self.asset else 'N/A'} at {self.time.date()}: ${self.price:.2f}>"