import importlib.util
from pathlib import Path
import secrets
from fastapi.testclient import TestClient

APP_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('easy_to_connect_app', APP_DIR / 'app.py')
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)

def test_health_and_public_city_catalog():
    client = TestClient(module.app)
    assert client.get('/api/health').status_code == 200
    cities = client.get('/api/cities')
    assert cities.status_code == 200
    assert any(x['name'] == 'Bhubaneswar' for x in cities.json())

def test_account_journey_budget_and_logout(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'DB_PATH', tmp_path / 'test.db')
    module.init_db()
    client = TestClient(module.app)
    email = f'{secrets.token_hex(5)}@example.com'
    registered = client.post('/api/auth/register', json={'email': email, 'password': 'InterviewPass123'})
    assert registered.status_code == 200
    assert client.get('/api/auth/me').status_code == 200
    comparison = client.get('/api/compare', params={'origin':'Cuttack','destination':'Bhubaneswar','budget_inr':7000})
    assert comparison.status_code == 200
    assert comparison.json()['results']
    option = comparison.json()['results'][0]
    saved = client.post('/api/plans', json={'origin':'Cuttack','destination':'Bhubaneswar','option':option,'reason':'Fastest for test journey'})
    assert saved.status_code == 200
    assert len(client.get('/api/plans').json()) == 1
    expense = client.post('/api/budget', json={'title':'Station auto','category':'Local transport','amount_inr':120})
    assert expense.status_code == 200
    assert client.get('/api/budget').json()['total_inr'] == 120
    assert client.post('/api/auth/logout').status_code == 200
    assert client.get('/api/auth/me').status_code == 401
    assert client.get('/api/plans').status_code == 401

def test_ml_predictor_returns_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'DB_PATH', tmp_path / 'predict.db')
    module.init_db()
    client = TestClient(module.app)
    email = f'{secrets.token_hex(5)}@example.com'
    assert client.post('/api/auth/register', json={'email': email, 'password': 'InterviewPass123'}).status_code == 200
    info = client.get('/api/model-info')
    assert info.status_code == 200
    assert info.json()['status'] == 'ready'
    result = client.post('/api/predict', json={
        'distance_km': 250, 'mode': 'train', 'transfers': 0,
        'departure_hour': 9, 'weather_factor': 0.35
    })
    assert result.status_code == 200
    assert result.json()['predicted_delay_minutes'] >= 0
    assert result.json()['delay_risk'] in {'Low', 'Moderate', 'High'}


def test_nearby_uses_bundled_fallback_when_overpass_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'DB_PATH', tmp_path / 'nearby.db')
    module.init_db()
    def offline(*args, **kwargs):
        raise module.requests.ConnectionError('offline test')
    monkeypatch.setattr(module.requests, 'post', offline)
    client = TestClient(module.app)
    email = f'{secrets.token_hex(5)}@example.com'
    assert client.post('/api/auth/register', json={'email': email, 'password': 'InterviewPass123'}).status_code == 200
    result = client.get('/api/nearby', params={'lat': 20.2707, 'lon': 85.8407})
    assert result.status_code == 200
    body = result.json()
    assert body['is_fallback'] is True
    assert body['places']
    assert 'demonstration' in body['source'].lower()


def test_user_data_is_private_between_accounts(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'DB_PATH', tmp_path / 'private-data.db')
    module.init_db()
    first = TestClient(module.app)
    second = TestClient(module.app)
    first_email = f'{secrets.token_hex(5)}@gmail.com'
    second_email = f'{secrets.token_hex(5)}@gmail.com'
    assert first.post('/api/auth/register', json={'email': first_email, 'password': 'InterviewPass123'}).status_code == 200
    assert second.post('/api/auth/register', json={'email': second_email, 'password': 'InterviewPass123'}).status_code == 200

    plan = first.post('/api/plans', json={
        'origin': 'Cuttack', 'destination': 'Bhubaneswar',
        'option': {'mode': 'Train', 'cost_inr': 120}, 'reason': 'Private test plan',
    })
    assert plan.status_code == 200
    assert len(first.get('/api/plans').json()) == 1
    assert second.get('/api/plans').json() == []

    expense = first.post('/api/budget', json={'title': 'Private auto fare', 'category': 'Local transport', 'amount_inr': 120})
    assert expense.status_code == 200
    assert first.get('/api/budget').json()['total_inr'] == 120
    assert second.get('/api/budget').json()['total_inr'] == 0

    foreign_plan_id = plan.json()['plan_id']
    assert second.delete(f'/api/plans/{foreign_plan_id}').status_code == 404
    foreign_expense_id = expense.json()['id']
    assert second.delete(f'/api/budget/{foreign_expense_id}').status_code == 404


def test_google_login_requires_client_id_when_not_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'DB_PATH', tmp_path / 'google-config.db')
    monkeypatch.delenv('GOOGLE_CLIENT_ID', raising=False)
    module.init_db()
    client = TestClient(module.app)
    assert client.get('/api/auth/google/config').json()['client_id'] == ''
    response = client.post('/api/auth/google', json={'credential': 'x' * 30})
    assert response.status_code == 503
