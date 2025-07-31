import sys
sys.path.append(r'C:\Users\User\Desktop\Data')

# === IMPORTS ===
from Preprocessing.preprocessing import load_and_clean_data
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from joblib import dump
import pandas as pd
import numpy as np
import os

# Pour visualisation interactive
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

# ============================
# 1. PREDICTION DU CHURN
# ============================
def churn_prediction_model():
    """Entraîne plusieurs modèles et retourne le meilleur selon accuracy"""
    df = load_and_clean_data()

    # Variable churn
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    last_date = df['InvoiceDate'].max()
    df['days_since_last_purchase'] = (last_date - df['InvoiceDate']).dt.days
    df['Churn'] = (df['days_since_last_purchase'] > 180).astype(int)

    # Encoder Country
    encoder = LabelEncoder()
    df['Country_encoded'] = encoder.fit_transform(df['Country'])

    # Features
    X = df[['Quantity', 'Price', 'Year', 'Customer_ID', 'Country_encoded']]
    y = df['Churn']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    results = {}
    best_model, best_score = None, 0

    for name, model in models.items():
        print(f"\n===== {name} =====")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"Accuracy: {acc:.4f}")

        # Visualisations uniquement pour Random Forest
        if name == "Random Forest":
            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            labels = ["Non Churn", "Churn"]

            fig_cm = ff.create_annotated_heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale='Blues',
                showscale=True
            )
            fig_cm.update_layout(title="Matrice de confusion - Random Forest")
            fig_cm.show()

            # Importance des variables
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                features = X.columns
                fig_imp = px.bar(x=features, y=importance,
                                 title="Importance des variables (Churn)")
                fig_imp.update_layout(xaxis_title="Variable", yaxis_title="Importance")
                fig_imp.show()

        if acc > best_score:
            best_score, best_model = acc, model

    print("\n=== Résumé des performances (Churn) ===")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")
    print(f"\n🏆 Meilleur modèle Churn : {best_model.__class__.__name__} (accuracy={best_score:.4f})")

    # Sauvegarde
    dump(best_model, "best_churn_model.joblib")
    print("Modèle churn sauvegardé : best_churn_model.joblib")

    return best_model, results


# ============================
# 2. MODELES DE PREVISION
# ============================

def forecast_sales_arima(df, n_months_to_predict=12, test_size=12):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['sales'] = df['Quantity'] * df['Price']
    monthly_sales = df.groupby(pd.Grouper(key='InvoiceDate', freq='M'))['sales'].sum()

    train, test = monthly_sales[:-test_size], monthly_sales[-test_size:]
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    pred = model_fit.forecast(steps=test_size)

    rmse = np.sqrt(mean_squared_error(test, pred))
    mae = mean_absolute_error(test, pred)
    mape = np.mean(np.abs((test - pred) / test)) * 100
    r2 = r2_score(test, pred)

    print("\n=== Métriques ARIMA (validation) ===")
    print(f"RMSE : {rmse:.2f}\nMAE  : {mae:.2f}\nMAPE : {mape:.2f}%\nR²   : {r2:.4f}")

    final_model = ARIMA(monthly_sales, order=(5, 1, 0)).fit()
    return final_model, (rmse, mae, mape, r2)


def forecast_sales_prophet(df, n_months_to_predict=12, test_size=12):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['sales'] = df['Quantity'] * df['Price']
    monthly_sales = df.groupby(pd.Grouper(key='InvoiceDate', freq='M'))['sales'].sum()

    prophet_df = monthly_sales.reset_index()
    prophet_df.columns = ['ds', 'y']
    train = prophet_df[:-test_size]
    test = prophet_df[-test_size:]

    model = Prophet()
    model.fit(train)
    future = model.make_future_dataframe(periods=test_size + n_months_to_predict, freq='M')
    forecast = model.predict(future)

    pred_test = forecast.set_index('ds').loc[test['ds'], 'yhat']

    rmse = np.sqrt(mean_squared_error(test['y'], pred_test))
    mae = mean_absolute_error(test['y'], pred_test)
    mape = np.mean(np.abs((test['y'] - pred_test) / test['y'])) * 100
    r2 = r2_score(test['y'], pred_test)

    print("\n=== Métriques Prophet (validation) ===")
    print(f"RMSE : {rmse:.2f}\nMAE  : {mae:.2f}\nMAPE : {mape:.2f}%\nR²   : {r2:.4f}")

    return model, (rmse, mae, mape, r2)


def forecast_sales_sarima(df, n_months_to_predict=12, seasonal_period=12):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['sales'] = df['Quantity'] * df['Price']
    monthly_sales = df.groupby(pd.Grouper(key='InvoiceDate', freq='M'))['sales'].sum()

    n_test = int(len(monthly_sales) * 0.2)
    train, test = monthly_sales[:-n_test], monthly_sales[-n_test:]

    # Modèle SARIMA entraîné sur les données d'entraînement
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,seasonal_period),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    # Prévisions sur la période test
    pred_mean = results.get_forecast(steps=n_test).predicted_mean

    rmse = np.sqrt(mean_squared_error(test, pred_mean))
    mae = mean_absolute_error(test, pred_mean)
    mape = np.mean(np.abs((test - pred_mean) / test)) * 100
    r2 = r2_score(test, pred_mean)

    print("\n=== Métriques SARIMA (validation) ===")
    print(f"RMSE : {rmse:.2f}\nMAE  : {mae:.2f}\nMAPE : {mape:.2f}%\nR²   : {r2:.4f}")

    # === Nouvelle prévision : entraînement sur toutes les données et projection future ===
    final_model = SARIMAX(monthly_sales, order=(1,1,1),
                          seasonal_order=(1,1,1,seasonal_period),
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)

    # Prévision future (après toute la série historique)
    future_forecast = final_model.get_forecast(steps=n_months_to_predict)
    future_mean = future_forecast.predicted_mean
    future_index = pd.date_range(start=monthly_sales.index[-1] + pd.offsets.MonthBegin(1),
                                 periods=n_months_to_predict, freq='M')

    # === VISUALISATION INTERACTIVE POUR SARIMA ===
    fig = go.Figure()
    # Historique complet
    fig.add_trace(go.Scatter(x=monthly_sales.index, y=monthly_sales,
                             mode='lines', name='Historique'))
    # Prévisions futures
    fig.add_trace(go.Scatter(x=future_index, y=future_mean,
                             mode='lines', name='Prévision SARIMA', line=dict(dash='dash')))

    fig.update_layout(
        title="Prévisions SARIMA avec projection future",
        xaxis_title="Date",
        yaxis_title="Ventes",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    fig.show()

    return results, (rmse, mae, mape, r2)


# ============================
# 3. COMPARAISON
# ============================

def compare_and_save_best_forecasting_model(df):
    # On entraîne quand même les trois modèles pour comparer, mais seul SARIMA affiche un graphique
    arima_model, arima_metrics = forecast_sales_arima(df)
    prophet_model, prophet_metrics = forecast_sales_prophet(df)
    sarima_model, sarima_metrics = forecast_sales_sarima(df)

    all_models = {
        "ARIMA": (arima_metrics[0], arima_model, arima_metrics[3]),
        "Prophet": (prophet_metrics[0], prophet_model, prophet_metrics[3]),
        "SARIMA": (sarima_metrics[0], sarima_model, sarima_metrics[3]),
    }

    # Choisir le meilleur selon le R²
    best_model_name = max(all_models, key=lambda k: all_models[k][2])
    best_r2 = all_models[best_model_name][2]
    best_model = all_models[best_model_name][1]

    print(f"\n🏆 Meilleur modèle forecasting : {best_model_name} avec R² = {best_r2:.4f}")

    model_filename = f"best_forecasting_model_{best_model_name}.joblib"
    dump(best_model, model_filename)
    print(f"Modèle forecasting sauvegardé sous : {model_filename}")


# ============================
# 4. MAIN
# ============================
if __name__ == "__main__":
    # Meilleur modèle churn
    churn_prediction_model()

    # Meilleur modèle forecasting
    df = load_and_clean_data()
    compare_and_save_best_forecasting_model(df)
