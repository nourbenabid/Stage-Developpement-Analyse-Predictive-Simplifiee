import sys
import os
sys.path.append(r'C:\Users\User\Desktop\Data')  # ajoute ton dossier racine au path

import pandas as pd
import pytest
from Preprocessing import preprocessing



# ---------- Tests unitaires pour load_and_clean_data ----------

def test_load_and_clean_data_returns_dataframe(monkeypatch):
    """
    Test que load_and_clean_data retourne bien un DataFrame même avec un MongoDB mocké
    """

    # Mock de la fonction MongoClient pour éviter d'accéder à une vraie DB
    class FakeCollection:
        def find(self, *args, **kwargs):
            return [
                {
                    "Customer_ID": 12345,
                    "Country": "France",
                    "Country_Code": "FR",
                    "Year": 2021,
                    "Price": 10.5,
                    "Quantity": 2,
                    "GDP": None,
                    "Inflation": 2.5,
                    "Consumption": 5.0,
                },
                {
                    "Customer_ID": None,
                    "Country": "Unknown",
                    "Country_Code": None,
                    "Year": 2021,
                    "Price": 3.0,
                    "Quantity": 1,
                    "GDP": 20,
                    "Inflation": None,
                    "Consumption": None,
                },
            ]

    class FakeDB:
        def __getitem__(self, name):
            return FakeCollection()

    class FakeClient:
        def __getitem__(self, name):
            return FakeDB()

    monkeypatch.setattr(preprocessing, "MongoClient", lambda uri: FakeClient())

    df = preprocessing.load_and_clean_data()

    # Vérifie que la sortie est un DataFrame
    assert isinstance(df, pd.DataFrame)

    # Vérifie qu'il reste au moins une ligne (la ligne avec Customer_ID non null)
    assert len(df) > 0

    # Vérifie que les colonnes essentielles sont présentes
    expected_cols = {"Customer_ID", "Country", "Price", "Quantity", "GDP"}
    assert expected_cols.issubset(df.columns)

def test_no_null_customer_ids(monkeypatch):
    """
    Test que la colonne Customer_ID ne contient pas de valeurs nulles après nettoyage.
    """
    # On utilise le même mock que précédemment
    class FakeCollection:
        def find(self, *args, **kwargs):
            return [
                {"Customer_ID": 12345, "Country": "France", "Country_Code": "FR", "Year": 2021,
                 "Price": 10.5, "Quantity": 2, "GDP": None, "Inflation": None, "Consumption": None},
                {"Customer_ID": None, "Country": "Test", "Country_Code": None, "Year": 2021,
                 "Price": 5.0, "Quantity": 1, "GDP": 10, "Inflation": 3, "Consumption": 4},
            ]

    class FakeDB:
        def __getitem__(self, name):
            return FakeCollection()

    class FakeClient:
        def __getitem__(self, name):
            return FakeDB()

    monkeypatch.setattr(preprocessing, "MongoClient", lambda uri: FakeClient())

    df = preprocessing.load_and_clean_data()

    # Vérifie qu'il n'y a pas de valeurs nulles dans Customer_ID
    assert df["Customer_ID"].isnull().sum() == 0
