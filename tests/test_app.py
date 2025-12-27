import pytest
import sys
import os

# On ajoute le dossier src au chemin pour que le test trouve l'app
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_homepage(client):
    """Test 1: Vérifier que la page d'accueil s'affiche"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Texelai" in response.data

def test_dashboard_access(client):
    """Test 2: Vérifier que le dashboard monitoring est accessible"""
    response = client.get('/dashboard')
    assert response.status_code == 200
    assert b"Dashboard" in response.data
