# ============================
# 📦 IMPORTATIONS DES LIBRAIRIES
# ============================
import os
from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv

# ============================
# 🔐 CHARGEMENT DES VARIABLES D’ENVIRONNEMENT
# ============================
# Permet de lire les paramètres sensibles (URI MongoDB, etc.) depuis un fichier .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

mongo_uri = os.getenv("MONGO_URI")
mongo_db = os.getenv("MONGO_DB")
mongo_collection = os.getenv("MONGO_COLLECTION")
def load_and_clean_data():
    try:
        # ============================
        # 📡 CONNEXION À LA BASE DE DONNÉES MONGODB
        # ============================
        client = MongoClient(mongo_uri)
        db = client[mongo_db]
        collection = db[mongo_collection]

        # ============================
        # 🧾 LECTURE DES DONNÉES DEPUIS MONGODB
        # ============================
        # `_id: 0` signifie qu'on exclut la colonne Mongo "_id" inutile ici
        cursor = collection.find({}, {"_id": 0})
        df = pd.DataFrame(list(cursor))

        print("✅ Données chargées depuis MongoDB.")
        print(df.head())

        # ============================
        # 🧼 NETTOYAGE DES DONNÉES
        # ============================

        # 🔎 Étape 1 : Suppression des lignes sans identifiant client
        lignes_avant = len(df)
        df = df.dropna(subset=["Customer_ID"])  # Les lignes sans client sont inutilisables
        df.reset_index(drop=True, inplace=True)
        lignes_apres = len(df)
        print(f"🧹 Lignes supprimées (Customer_ID manquant) : {lignes_avant - lignes_apres}")

        # 🔎 Étape 2 : Affichage des valeurs manquantes restantes
        print("\n🚨 Valeurs manquantes par colonne :")
        print(df.isnull().sum())


        
        # 🔎 Étape 3 : Vérification des Country_Code manquants
        missing_country_code = df[df['Country_Code'].isnull()]['Country'].unique()
        print("🌍 Pays sans code pays :", missing_country_code)

        # 🚮 Étape 4 : Suppression des lignes dont le Country_Code est manquant
        # En général, cela correspond aux lignes avec un pays "Unspecified"
        lignes_avant = len(df)
        df = df[df['Country_Code'].notna()]
        lignes_apres = len(df)
        print(f"🗑️ Lignes supprimées (Country_Code manquant) : {lignes_avant - lignes_apres}")

        # 🔄 Étape 5 : Réaffichage des valeurs manquantes restantes
        print(df.isnull().sum())

        # ============================
        # 🔁 IMPUTATION DES INDICATEURS ÉCONOMIQUES
        # ============================
        colonnes_a_imputer = ['GDP', 'Inflation', 'Consumption']

        # Imputation par médiane, groupée par année uniquement
        # Pourquoi pas (Country_Code, Year) ici ? pour éviter trop de petits groupes
        for col in colonnes_a_imputer:
            df[col] = df.groupby(['Year'])[col].transform(lambda x: x.fillna(x.median()))

        # 🔍 Résultat final : combien de valeurs restent manquantes ?
        print("📊 Valeurs manquantes après imputation :")
        print(df.isnull().sum())
        return df

    except Exception as e:
        print(f"❌ Erreur lors du chargement ou nettoyage : {e}")
        return pd.DataFrame()  # retourne un DataFrame vide si erreur
if __name__ == "__main__":
 df = load_and_clean_data()
 print("✅ Fonction exécutée avec succès")
