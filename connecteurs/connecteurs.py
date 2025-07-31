import os
import time
import requests
import pandas as pd
from pymongo import MongoClient
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv

# === CONFIGURATION ===
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

INDICATORS = {
    "GDP": "NY.GDP.MKTP.CD",
    "Inflation": "FP.CPI.TOTL.ZG",
    "Consumption": "NE.CON.TOTL.ZS"
}
WB_API_URL = "http://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={year}&format=json"

# === KAGGLE CONNECTEUR ===
def init_kaggle_api():
    api = KaggleApi()
    api.authenticate()
    return api

def download_and_extract_dataset(api, dataset_slug, dest_folder="data/"):
    os.makedirs(dest_folder, exist_ok=True)
    print(f" Téléchargement du dataset : {dataset_slug}")
    api.dataset_download_files(dataset_slug, path=dest_folder, unzip=True)
    print(f" Dataset extrait dans : {dest_folder}")

def find_main_csv_file(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            return os.path.join(folder_path, file)
    raise FileNotFoundError(" Aucun fichier .csv trouvé dans le dossier.")

def save_clean_csv(df, output_path="data/online_retail.csv"):
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    df.to_csv(output_path, index=False)
    print(f" Fichier final sauvegardé : {output_path}")

def delete_other_csvs(folder_path, keep_filename="online_retail.csv"):
    for file in os.listdir(folder_path):
        if file.endswith(".csv") and file != keep_filename:
            os.remove(os.path.join(folder_path, file))
            print(f" Fichier temporaire supprimé : {file}")

# === ENRICHISSEMENT ECONOMIQUE  ===
def get_economic_data(country_code, year):
    result = {}
    for label, indicator in INDICATORS.items():
        url = WB_API_URL.format(country=country_code.lower(), indicator=indicator, year=year)
        try:
            r = requests.get(url)
            if r.status_code == 200:
                data = r.json()
                value = data[1][0]['value'] if data and data[1] else None
                result[label] = value
            else:
                result[label] = None
        except:
            result[label] = None
        time.sleep(0.01)
    return result

def enrich_with_economic_indicators(df, date_col="InvoiceDate", country_col="Country"):
    print(" Enrichissement économique...")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['Year'] = df[date_col].dt.year

    country_map = {
    "United Kingdom": "GBR",
    "France": "FRA",
    "USA": "USA",
    "Belgium": "BEL",
    "Australia": "AUS",
    "EIRE": "IRL",
    "Germany": "DEU",
    "Portugal": "PRT",
    "Japan": "JPN",
    "Denmark": "DNK",
    "Netherlands": "NLD",
    "Poland": "POL",
    "Spain": "ESP",
    "Italy": "ITA",
    "Cyprus": "CYP",
    "Greece": "GRC",
    "Norway": "NOR",
    "Austria": "AUT",
    "Sweden": "SWE",
    "United Arab Emirates": "ARE",
    "Finland": "FIN",
    "Switzerland": "CHE",
    "Nigeria": "NGA",
    "Malta": "MLT",
    "RSA": "ZAF",
    "Singapore": "SGP",
    "Bahrain": "BHR",
    "Thailand": "THA",
    "Israel": "ISR",
    "Lithuania": "LTU",
    "West Indies": "WST",   # Ce code n’est pas officiel, à adapter
    "Korea": "KOR",
    "Brazil": "BRA",
    "Canada": "CAN",
    "Iceland": "ISL",
    "Lebanon": "LBN",
    "Saudi Arabia": "SAU",
    "Czech Republic": "CZE",
    "European Community": "EU",  # À adapter si besoin
    "Channel Islands": "CHI",    # Non ISO, à adapter
    
}


    df['Country_Code'] = df[country_col].map(country_map)

    enrichments = {}
    for _, row in df[["Year", "Country_Code"]].dropna().drop_duplicates().iterrows():
        year = int(row['Year'])
        code = row['Country_Code']
        enrichments[(year, code)] = get_economic_data(code, year)

    for label in INDICATORS:
        df[label] = df.apply(lambda row: enrichments.get((row['Year'], row['Country_Code']), {}).get(label), axis=1)

    print(" Enrichissement économique terminé.")
    return df

# === STOCKAGE MONGO ===
def store_in_mongo(df):
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        records = df.to_dict(orient="records")
        collection.delete_many({})
        collection.insert_many(records)
        print(f" Données stockées dans MongoDB ({MONGO_DB}.{MONGO_COLLECTION})")
    except Exception as e:
        print(" Erreur MongoDB :", e)

# === CONNECTEUR PRINCIPAL ===
def kaggle_connecteur_csv():
    dataset_slug = "mashlyn/online-retail-ii-uci"
    dest_folder = "data/"
    output_filename = "online_retail.csv"

    api = init_kaggle_api()
    download_and_extract_dataset(api, dataset_slug, dest_folder)

    csv_path = find_main_csv_file(dest_folder)
    df = pd.read_csv(csv_path, encoding='latin1')

    df = enrich_with_economic_indicators(df, date_col="InvoiceDate", country_col="Country")

    save_clean_csv(df, output_path=os.path.join(dest_folder, output_filename))
    delete_other_csvs(dest_folder, keep_filename=output_filename)

    store_in_mongo(df)
    return df

# === LANCEMENT ===
if __name__ == "__main__":
    df = kaggle_connecteur_csv()
    print(df.head())
