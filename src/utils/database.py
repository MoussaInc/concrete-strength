import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from dotenv import load_dotenv

# Chargement des variables d'environnement depuis le fichier .env
load_dotenv()

# Définition de l'URL de la BDD pour la production ou le développement
# Render définit la variable d'environnement DATABASE_URL automatiquement. En local, on construit l'URL à partir du .env.
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    pg_host = os.environ.get("PG_HOST")
    pg_port = os.environ.get("PG_PORT")
    pg_user = os.environ.get("PG_USER")
    pg_password = os.environ.get("PG_PASSWORD")
    pg_database = os.environ.get("PG_DATABASE")
    
    if not all([pg_host, pg_port, pg_user, pg_password, pg_database]):
        raise ValueError("Les variables de BDD ne sont pas configurées dans le .env ou dans l'environnement.")
        
    DATABASE_URL = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"

# Création du moteur de BDD
engine = create_engine(DATABASE_URL)
# Création de la base
Base = declarative_base()

# Définition du modèle de la table
class UsageLog(Base):
    __tablename__ = "usage_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    endpoint = Column(String)
    user_type = Column(String) # 'API' ou 'Dashboard'
    ip_address = Column(String, nullable=True) # Pour identification des utilisateurs

# Créer la table dans la base de données
def create_tables():
    Base.metadata.create_all(bind=engine)
    print("Tables de BDD créées ou déjà existantes.")

# Création d'une session de BDD
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dépendance pour FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    create_tables()