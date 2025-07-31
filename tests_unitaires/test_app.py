import sys
sys.path.append(r'C:\Users\User\Desktop\Data')  # ajoute ton dossier racine au path
import pytest
from api import app  # importe ton app Flask

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Country" in response.data  # adapte selon ton template

def test_forecast(client):
    response = client.get('/forecast')
    assert response.status_code == 200
    assert "Prévisions SARIMA" in response.data.decode('utf-8')


def test_predict_churn_form(client):
    data = {
        "Quantity": "10",
        "Price": "20.5",
        "Year": "2024",
        "Customer_ID": "12345",
        "Country": "France"  # Assure-toi que ce pays est dans ta liste countries
    }
    response = client.post('/predict_churn_form', data=data, follow_redirects=True)
    assert response.status_code == 200
    assert b"Churn" in response.data or b"Non Churn" in response.data

def test_download_forecast_excel(client):
    response = client.get('/download_forecast_excel')
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    assert 'attachment; filename=previsions_sarima.xlsx' in response.headers['Content-Disposition']
