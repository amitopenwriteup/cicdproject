import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app import app

def test_index():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
