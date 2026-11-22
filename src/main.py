from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, text
import uvicorn

app = FastAPI(title="HorizonForge")

app.mount("/static", StaticFiles(directory="static"), name="static")

# SQLite
engine = create_engine("sqlite:///horizonforge.db", echo=False)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    try:
        with open("templates/index.html", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except Exception as e:
        print("Ошибка загрузки шаблона:", e)
        return HTMLResponse("<h1>Ошибка загрузки дашборда</h1>")

@app.get("/api/assets")
async def get_assets():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT id, name, ticker FROM assets ORDER BY id"))
            assets = [dict(row._mapping) for row in result]
            print(f"✅ API /api/assets вернул {len(assets)} монет")
            return assets
    except Exception as e:
        print("❌ Ошибка в /api/assets:", e)
        return []

@app.get("/api/prices/{asset_id}")
async def get_prices(asset_id: int, limit: int = 60):
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT time, price, open, high, low, volume
                FROM market_prices 
                WHERE asset_id = :asset_id 
                ORDER BY time DESC 
                LIMIT :limit
            """), {"asset_id": asset_id, "limit": limit})
            prices = [dict(row._mapping) for row in result]
            print(f"✅ Цены для asset_id={asset_id}: {len(prices)} записей")
            return prices
    except Exception as e:
        print("❌ Ошибка в /api/prices:", e)
        return []

if __name__ == "__main__":
    print("🌐 Запуск HorizonForge (SQLite версия)...")
    uvicorn.run(app, host="0.0.0.0", port=8001)