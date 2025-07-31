import os
from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les variables d'environnement
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
# 🔐 Connexion à MongoDB
mongo_uri = os.getenv("MONGO_URI")
mongo_db = os.getenv("MONGO_DB")
mongo_collection = os.getenv("MONGO_COLLECTION")

try:
    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    collection = db[mongo_collection]

    #  Importer les données dans un DataFrame
    cursor = collection.find()
    df = pd.DataFrame(list(cursor))

   

    print("OK - Données importées depuis MongoDB :")

    print(df.head())

except Exception as e:
    print(f"Erreur lors de l'import : {e}")




# Data_Discovery# ----------------------------- #
# 📊 Analyse exploratoire des données (EDA)

# ----------------------------- #
# 🔍 Aperçu général du dataset
# ----------------------------- #
print(" Aperçu des premières lignes :")
print(df.head())

print("\n Informations sur les colonnes :")
print(df.info())  # types des colonnes + nb de valeurs non nulles

print("\n Dimensions du dataset :")
print(f"{df.shape[0]} lignes et {df.shape[1]} colonnes")

# ----------------------------- #
# 🔎 Statistiques descriptives
# ----------------------------- #
print("\n Statistiques descriptives :")
print(df.describe(include='all'))

# ----------------------------- #
# 🧼 Vérification des valeurs manquantes
# ----------------------------- #
print("\n Valeurs manquantes par colonne :")
print(df.isnull().sum())

#  Visualisation graphique des valeurs manquantes
sns.heatmap(df.isnull(), cbar=False, cmap='Reds')
plt.title("Carte des valeurs manquantes")
plt.show()

# ----------------------------- #
# 📋 Valeurs uniques (utile pour les colonnes catégorielles)
# ----------------------------- #
print("\n Nombre de valeurs uniques par colonne :")
print(df.nunique())

# Exemple : Voir les valeurs uniques de la colonne 'Country'
if 'Country' in df.columns:
    print("\n Liste des pays dans les données :")
    print(df['Country'].unique())

# ----------------------------- #
#  Détection et affichage des doublons
duplicate_rows = df[df.duplicated()]
print(f"\n Nombre de doublons : {duplicate_rows.shape[0]}")
print("\n Lignes dupliquées :")
print(duplicate_rows)
# Détection des doublons exacts sur toutes les colonnes
exact_duplicates = df[df.duplicated(keep=False)]

print(f" Nombre de doublons stricts (lignes identiques sur toutes les colonnes) : {exact_duplicates.shape[0]}")

# Affichage des premières lignes dupliquées exactes
print(" Lignes strictement identiques :")
print(exact_duplicates.head(10))



# 🔍 Détection des outliers par la méthode de l'écart interquartile (IQR)
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"\n Valeurs aberrantes pour '{column}' : {outliers.shape[0]} lignes")
    return outliers

# Exemple pour 'Quantity' et 'Price'
outliers_quantity = detect_outliers_iqr(df, 'Quantity')
print(outliers_quantity[['Invoice', 'Quantity']].head())

outliers_price = detect_outliers_iqr(df, 'Price')
print(outliers_price[['Invoice', 'Price']].head())



# ----------------------------- #
# 📦 Analyse d'une variable clé (ex: Quantity, UnitPrice)
# ----------------------------- #
if 'Quantity' in df.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Quantity'], bins=50, kde=True)
    plt.title("Distribution des Quantités")
    plt.show()

if 'UnitPrice' in df.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df['UnitPrice'], bins=50, kde=True)
    plt.title("Distribution des Prix Unitaires")
    plt.show()
# ----------------------------- #
# 🔗 Matrice de corrélation
# ----------------------------- #
plt.figure(figsize=(12, 8))
numerical_cols = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_cols.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title(" Matrice de Corrélation des Variables Numériques")
plt.show()


# ----------------------------- #
# 🧮 Analyse temporelle si la colonne 'InvoiceDate' existe
# ----------------------------- #
if 'InvoiceDate' in df.columns:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['Month'] = df['InvoiceDate'].dt.month
    df['Year'] = df['InvoiceDate'].dt.year

    plt.figure(figsize=(10, 4))
    df['Month'].value_counts().sort_index().plot(kind='bar')
    plt.title("Transactions par Mois")
    plt.xlabel("Mois")
    plt.ylabel("Nombre de transactions")
    plt.show()
