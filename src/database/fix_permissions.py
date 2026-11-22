from sqlalchemy import create_engine, text

print("🔧 Исправляем права доступа...")

# Подключаемся как postgres (суперпользователь)
engine = create_engine(
    "postgresql+psycopg2://postgres@localhost:5432/horizonforge"
    # Если у тебя есть пароль у postgres, добавь его сюда:
    # "postgresql+psycopg2://postgres:ТВОЙ_ПАРОЛЬ@localhost:5432/horizonforge"
)

try:
    with engine.connect() as conn:
        conn.execute(text("GRANT ALL PRIVILEGES ON SCHEMA public TO horizonforge;"))
        conn.execute(text("GRANT ALL ON ALL TABLES IN SCHEMA public TO horizonforge;"))
        conn.execute(text("GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO horizonforge;"))
        conn.execute(text("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO horizonforge;"))
        conn.execute(text("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO horizonforge;"))
        
        print("✅ Права успешно выданы пользователю horizonforge!")
        print("Теперь можно создавать таблицы.")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    print("\nЕсли ошибка с паролем — скажи мне, и мы укажем пароль postgres.")