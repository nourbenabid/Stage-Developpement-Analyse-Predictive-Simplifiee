
import sys, os
# Ajoute le dossier parent (Data) au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import io
from contextlib import redirect_stdout
from flask import Flask, render_template, request
from joblib import load
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objs as go
from plotly.offline import plot
from flask import make_response
from io import BytesIO
base_dir = os.path.dirname(__file__)




# Ajouter le dossier parent au sys.path

from Preprocessing.preprocessing import load_and_clean_data

app = Flask(__name__)

# ==============================
# Chargement des données propres
# ==============================
f = io.StringIO()
with redirect_stdout(f):
    df = load_and_clean_data()

# Encodage pays
countries = sorted(df['Country'].unique())
encoder = LabelEncoder()
encoder.fit(df['Country'])

# ==============================
# Chargement des modèles
# ==============================
churn_model_path = os.path.abspath(os.path.join(base_dir, '..', 'best_churn_model.joblib'))
forecasting_model_path = os.path.abspath(os.path.join(base_dir, '..', 'best_forecasting_model_SARIMA.joblib'))

churn_model = load(churn_model_path)
forecasting_model = load(forecasting_model_path)

# Données agrégées pour forecasting
df_forecast = df.copy()
df_forecast['InvoiceDate'] = pd.to_datetime(df_forecast['InvoiceDate'])
df_forecast['sales'] = df_forecast['Quantity'] * df_forecast['Price']
monthly_sales = df_forecast.groupby(pd.Grouper(key='InvoiceDate', freq='M'))['sales'].sum()


# ==============================
# Routes Flask
# ==============================
@app.route('/')
def index():
    return render_template('index.html', countries=countries)

@app.route('/predict_churn_form', methods=['POST'])
def predict_churn_form():
    # Récupération des données formulaire
    quantity = float(request.form['Quantity'])
    price = float(request.form['Price'])
    year = int(request.form['Year'])
    customer_id = int(request.form['Customer_ID'])
    country = request.form['Country']

    # Encoder le pays choisi
    country_encoded = encoder.transform([country])[0]

    # Construire X
    X = pd.DataFrame([{
        "Quantity": quantity,
        "Price": price,
        "Year": year,
        "Customer_ID": customer_id,
        "Country_encoded": country_encoded
    }])

    # Prédiction
    prediction = churn_model.predict(X)[0]
    prob = None
    if hasattr(churn_model, "predict_proba"):
        prob = round(churn_model.predict_proba(X)[0][1] * 100, 2)

    result = "Churn" if prediction == 1 else "Non Churn"

    return render_template(
        'index.html',
        countries=countries,
        prediction_result=result,
        prediction_prob=prob
    )

@app.route('/forecast')
def forecast():
    n_months_to_predict = 12
    forecast_results = forecasting_model.get_forecast(steps=n_months_to_predict)
    forecast_mean = forecast_results.predicted_mean
    future_index = pd.date_range(
        start=monthly_sales.index[-1] + pd.offsets.MonthBegin(1),
        periods=n_months_to_predict,
        freq='M'
    )

    # Préparer les données
    forecast_data = list(zip(future_index.strftime("%Y-%m"), forecast_mean.round(2)))

    # Graphique Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_sales.index, y=monthly_sales, mode='lines', name='Historique'))
    fig.add_trace(go.Scatter(x=future_index, y=forecast_mean, mode='lines', name='Prévision SARIMA', line=dict(dash='dash')))
    fig.update_layout(title="Prévisions SARIMA", xaxis_title="Date", yaxis_title="Ventes")
    graph_html = plot(fig, output_type='div', include_plotlyjs=True)

    return render_template('forecast.html', forecast_data=forecast_data, graph_html=graph_html)
# Nouvelle route : téléchargement Excel
@app.route('/download_forecast_excel')
def download_forecast_excel():
    n_months_to_predict = 12
    forecast_results = forecasting_model.get_forecast(steps=n_months_to_predict)
    forecast_mean = forecast_results.predicted_mean
    future_index = pd.date_range(
        start=monthly_sales.index[-1] + pd.offsets.MonthBegin(1),
        periods=n_months_to_predict,
        freq='M'
    )

    forecast_df = pd.DataFrame({
        'Mois': future_index.strftime("%Y-%m"),
        'Ventes prévues': forecast_mean.round(2)
    })

    # Sauvegarde dans un buffer
    output = BytesIO()
    forecast_df.to_excel(output, index=False)
    output.seek(0)

    response = make_response(output.read())
    response.headers['Content-Disposition'] = 'attachment; filename=previsions_sarima.xlsx'
    response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    return response


if __name__ == "__main__":
    app.run(debug=True)
