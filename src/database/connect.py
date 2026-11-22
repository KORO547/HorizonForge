import psycopg2
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime

# Параметры подключения
DB_PARAMS = {
    "dbname": "horizonforge",
    "user": "horizonforge",
    "password": "horizon123",
    "host": "localhost",
    "port": "5432"
}

# Подключение через psycopg2
def test_connection():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        print("✅ Успешное подключение к базе!")
        
        # Проверяем таблицы
        cur = conn.cursor()
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = cur.fetchall()
        print("Таблицы в базе:")
        for table in tables:
            print(f"   • {table[0]}")
        
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

if __name__ == "__main__":
    test_connection()