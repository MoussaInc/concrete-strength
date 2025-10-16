# src/utils/database.py

import os
import datetime
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

_ENGINE = None
_SessionLocal = None

Base = declarative_base()

def _normalize_pg_url(url: str) -> str:
    if not url:
        return url
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    try:
        p = urlparse(url)
        q = dict(parse_qsl(p.query))
        q.setdefault("sslmode", "require")
        url = urlunparse(p._replace(query=urlencode(q)))
    except Exception:
        pass
    return url

def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        host = os.getenv("PG_HOST", "127.0.0.1")
        port = int(os.getenv("PG_PORT", 5432))
        user = os.getenv("PG_USER", "mballo")
        pwd  = os.getenv("PG_PASSWORD", "supersecretpassword")
        db   = os.getenv("PG_DATABASE", "concrete_db")
        url = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    return _normalize_pg_url(url)

def _ensure_engine():
    """
    Crée engine + SessionLocal à la demande. Ne lève pas d'exception bloquante.
    """
    global _ENGINE, _SessionLocal
    if _ENGINE is not None and _SessionLocal is not None:
        return _ENGINE, _SessionLocal
    try:
        url = _db_url()
        _ENGINE = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_size=2,
            max_overflow=0,
            connect_args={"connect_timeout": 5},
        )
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_ENGINE)
        return _ENGINE, _SessionLocal
    except Exception as e:
        print(f"[DB] Engine init failed (non-fatal): {e}")
        _ENGINE, _SessionLocal = None, None
        return None, None

# --- Modèle ---
class UsageLog(Base):
    __tablename__ = "usage_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    endpoint = Column(String)
    user_type = Column(String)
    ip_address = Column(String)
    user_id = Column(String, index=True, nullable=False)

# --- Best-effort create ---
def create_tables():
    engine, _ = _ensure_engine()
    if engine is None:
        print("[DB] create_tables skipped: engine unavailable")
        return
    try:
        Base.metadata.create_all(bind=engine)
        print("[DB] Tables created or already exist.")
    except Exception as e:
        print(f"[DB] create_tables failed: {e}")

# --- Dépendance FastAPI (best-effort) ---
def get_db():
    engine, SessionLocal = _ensure_engine()
    if SessionLocal is None:
        class _Null:
            def __getattr__(self, name): raise AttributeError(name)
        db = _Null()
        try:
            yield db
        finally:
            pass
        return

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    create_tables()
