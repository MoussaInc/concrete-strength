# generate_test_csv.py
import pandas as pd

# Définir un exemple de données
data = {
    "cement": [200, 250, 300, 350, 400, 500],
    "slag": [0, 1, 2, 3, 4, 5],
    "fly_ash": [0, 1, 2, 3, 4, 5],
    "water": [80, 100, 120, 140, 160, 180],
    "superplasticizer": [0, 1, 2, 3, 4, 5],
    "coarse_aggregate": [1050, 950, 850, 750, 650, 550],
    "fine_aggregate": [450, 550, 650, 750, 850, 950],
    "age": [1, 2, 7, 14, 28, 90],
}

# Créer le DataFrame
df = pd.DataFrame(data)

# Sauvegarder dans un CSV
df.to_csv("data/test_input.csv", index=False)
print("Fichier CSV généré : data/test_input.csv")
