import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    pg_host = os.getenv("PG_HOST", "127.0.0.1")
    pg_port = int(os.getenv("PG_PORT", 5432))
    pg_user = os.getenv("PG_USER", "mballo")
    pg_password = os.getenv("PG_PASSWORD", "supersecretpassword")
    pg_database = os.getenv("PG_DATABASE", "concrete_db")
    DATABASE_URL = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"

# Création du moteur SQLAlchemy
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# -------------------------------
# Modèle UsageLog
# -------------------------------
class UsageLog(Base):
    __tablename__ = "usage_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    endpoint = Column(String)
    user_type = Column(String)
    ip_address = Column(String)
    user_id = Column(String, index=True, nullable=False)

# -------------------------------
# Création des tables
# -------------------------------
def create_tables():
    Base.metadata.create_all(bind=engine)
    print("Tables de BDD créées ou déjà existantes.")

# -------------------------------
# Session SQLAlchemy
# -------------------------------
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------
# Lancement directement pour creation des tables
# -------------------------------
if __name__ == "__main__":
    create_tables()
