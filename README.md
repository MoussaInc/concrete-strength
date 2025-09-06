

## 🏗️ Concrete Strength Predictor : Une application MLOps de bout en bout 🧱

Ce projet met en œuvre un pipeline complet de Machine Learning, de la gestion des données à l'exposition d'un modèle via une **API robuste et un dashboard interactif**. 
Il démontre une maîtrise des technologies modernes pour le déploiement de modèles de Machine Learning en production.

-----

## 🎯 Points Forts Techniques

Ce projet met en lumière plusieurs compétences clés pour les rôles de Data Scientist, MLOps Engineer ou Data Engineer :

  * **Ingénierie de données (ETL)** : Un pipeline d'ingestion et de nettoyage des données est mis en place pour transformer les données brutes de la composition du béton et les charger dans une base de données **PostgreSQL**.
  * **Modélisation prédictive** : Le projet inclut l'entraînement et la sélection d'un modèle **XGBoost** de pointe pour prédire la résistance du béton avec une grande précision.
  * **Développement API avec FastAPI** : Une API **FastAPI** est construite pour servir les prédictions du modèle de manière performante et évolutive, supportant les requêtes individuelles et en lot.
  * **Interface Utilisateur avec Streamlit** : Un dashboard **Streamlit** offre une interface utilisateur conviviale, simplifiant l'interaction avec le modèle pour les utilisateurs non techniques.
  * **Conteneurisation & Orchestration (Docker)** : L'ensemble de l'application est **conteneurisé à l'aide de Docker et orchestré avec Docker Compose**. Cela garantit la portabilité, la reproductibilité de l'environnement et facilite grandement le déploiement sur n'importe quel serveur.

-----

## 🚀 Démarrage Rapide

Pour lancer l'application en quelques commandes seulement.

### Prérequis

  * **Docker et Docker Compose** installés sur votre système.
  * Un fichier `.env` à la racine du projet pour configurer l'accès à la base de données (voir exemple ci-dessous).

<!-- end list -->

```env
POSTGRES_DB=concrete_db
POSTGRES_USER=concrete_user
POSTGRES_PASSWORD=your_password
```

### Lancement du Projet

1.  Clonez le dépôt :

    ```bash
    git clone https://github.com/MoussaInc/concrete-strength.git
    ```

2.  Naviguez dans le répertoire du projet :

    ```bash
    cd concrete-strength-predictor
    ```

3.  Lancez les services Docker :

    ```bash
    docker-compose up --build
    ```

L'API sera disponible à l'adresse **`http://localhost:8000`** et le dashboard Streamlit à **`http://localhost:8501`**.

-----

## 📂 Structure du Projet

```
.
├── config
├── data
├── docker
│   ├── api
│   └── dashboard
├── docker-compose.yml
├── LICENSE
├── models
├── notebooks
├── README.md
├── requirements.txt
├── script
│   └── run_app.sh
└── src
    ├── api
    ├── dashboard
    ├── etl
    ├── ml
    └── utils
```

-----

## 📚 En savoir plus

Pour une description plus détaillée du projet, de la méthodologie et des choix technologiques, veuillez consulter la documentation dans le dossier `notebooks/` et le rapport d'analyse.