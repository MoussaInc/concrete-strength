# src/utils/database.py

import os
import datetime
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

try:
    import psycopg 
    _PSYCOPG3 = True
except Exception:
    _PSYCOPG3 = False

# ------------- Helpers URL ------------- #
def _normalize_pg_url(url: str) -> str:
    """
    - Convertit 'postgres://' -> 'postgresql://'
    - Ajoute le driver si psycopg3 dispo: 'postgresql+psycopg://'
    - Force sslmode=require si absent (utile sur Render)
    """
    if not url:
        return url
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if _PSYCOPG3 and url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    try:
        parsed = urlparse(url)
        q = dict(parse_qsl(parsed.query))
        if "sslmode" not in q and parsed.scheme.startswith("postgresql"):
            q["sslmode"] = "require"
        new_query = urlencode(q)
        url = urlunparse(parsed._replace(query=new_query))
    except Exception:
        pass

    return url

# ------------- Configuration ------------- #
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    pg_host = os.getenv("PG_HOST", "127.0.0.1")
    pg_port = int(os.getenv("PG_PORT", 5432))
    pg_user = os.getenv("PG_USER", "mballo")
    pg_password = os.getenv("PG_PASSWORD", "supersecretpassword")
    pg_database = os.getenv("PG_DATABASE", "concrete_db")
    DATABASE_URL = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"

DATABASE_URL = _normalize_pg_url(DATABASE_URL)
ENGINE_KW = dict(
    pool_pre_ping=True,
    pool_recycle=1800,    
    pool_size=2,
    max_overflow=0,       
    connect_args={"connect_timeout": 5},
)

engine = create_engine(DATABASE_URL, **ENGINE_KW)
Base = declarative_base()

# ------------- Modèles ------------- #
class UsageLog(Base):
    __tablename__ = "usage_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    endpoint = Column(String)
    user_type = Column(String)
    ip_address = Column(String)
    user_id = Column(String, index=True, nullable=False)

# ------------- Création tables (best effort) ------------- #
def create_tables():
    """
    À appeler au boot (best-effort).
    Ne doit PAS faire planter l'app si la DB n'est pas prête.
    """
    try:
        Base.metadata.create_all(bind=engine)
        print("[DB] Tables created or already exist.")
    except Exception as e:
        print(f"[DB] create_tables skipped/failed: {e}")

# ------------- Session ------------- #
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Renvoie une session SQLAlchemy.
    Note: la connexion réelle ne se fait qu'à la première requête SQL.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    create_tables()
