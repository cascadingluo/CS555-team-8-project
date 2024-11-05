import sys
from unittest.mock import MagicMock

# Mock google.generativeai to prevent it from causing import issues
sys.modules["google.generativeai"] = MagicMock()
import pytest
from app import app  # Import the Flask app instance from app.py
import json

@pytest.fixture
def client():
    """Setup test client for Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def mock_chat_response(input_text):
    """Return a mock response that simulates a realistic JSON response from the chat API."""
    return {"text": f"Mock response to: {input_text}"}

# Tests for Login Component functionality
def test_login_form_rendered_correctly(client):
    """Test if the login page renders correctly"""
    response = client.get('/login')  # Assuming there is a login route
    assert response.status_code == 200
    assert b"username" in response.data
    assert b"password" in response.data
    assert b"submit" in response.data

def test_user_can_type_username_password(client):
    """Test login functionality by sending POST request with username and password"""
    # Simulate submitting the login form
    response = client.post('/login', data={
        'username': 'testUser',
        'password': 'testPassword'
    })
    assert response.status_code in [200, 302] 
    if response.status_code == 200:
        assert b"Welcome" in response.data 
