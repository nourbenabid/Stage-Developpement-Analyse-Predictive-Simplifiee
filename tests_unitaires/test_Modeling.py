import sys
import os
sys.path.append(r'C:\Users\User\Desktop\Data')  # ajoute ton dossier racine au path
import pandas as pd
import numpy as np
from joblib import load
from Modéles_Machine_Learning import Modeling
import os

# --------- Dataset factice ---------
def fake_df():
    dates = pd.date_range("2021-01-01", periods=24, freq="M")
    return pd.DataFrame({
        "InvoiceDate": np.tile(dates, 2),
        "Quantity": np.random.randint(1, 5, size=48),
        "Price": np.random.uniform(10, 50, size=48),
        "Customer_ID": np.arange(48),
        "Country": ["France"] * 48,
        "Country_Code": ["FR"] * 48,
        "Year": [2021] * 48,
        "GDP": [10] * 48,
        "Inflation": [2] * 48,
        "Consumption": [5] * 48,
    })

# --------- Test churn_prediction_model ---------
def test_churn_prediction_model(monkeypatch, tmp_path):
    # Mock de load_and_clean_data pour éviter Mongo
    monkeypatch.setattr(Modeling, "load_and_clean_data", lambda: fake_df())

    # Change répertoire courant pour sauvegarder dans tmp_path
    old_cwd = os.getcwd()
    os.chdir(tmp_path)

    best_model, results = Modeling.churn_prediction_model()

    # Vérif retour
    assert best_model is not None
    assert isinstance(results, dict)
    assert "Random Forest" in results

    # Vérifie que le fichier joblib est créé
    assert os.path.exists("best_churn_model.joblib")

    os.chdir(old_cwd)

# --------- Test forecasting avec SARIMA ---------
def test_forecast_sales_sarima():
    df = fake_df()
    model, metrics = Modeling.forecast_sales_sarima(df, n_months_to_predict=3, seasonal_period=6)

    assert model is not None
    assert isinstance(metrics, tuple)
    assert len(metrics) == 4  # rmse, mae, mape, r2

# Idem pour ARIMA et Prophet
def test_forecast_sales_arima():
    df = fake_df()
    model, metrics = Modeling.forecast_sales_arima(df, n_months_to_predict=3, test_size=3)
    assert model is not None

def test_forecast_sales_prophet():
    df = fake_df()
    model, metrics = Modeling.forecast_sales_prophet(df, n_months_to_predict=3, test_size=3)
    assert model is not None
