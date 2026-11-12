import uuid
from fastapi.testclient import TestClient
from main import app

# --- Create a test client (fake browser for testing) ---
client = TestClient(app)

def unique_email():
    """Generate a unique email for testing so runs don't conflict"""
    return f"user_{uuid.uuid4().hex[:8]}@example.com"

# -----------------------------------------------
# AUTH TESTS
# -----------------------------------------------

def test_register():
    """Test: Register a new user"""
    email = unique_email()
    response = client.post("/register", json={
        "email": email,
        "password": "test1234"
    })
    # Should return 200 OK
    assert response.status_code == 200
    assert response.json() == {"message": "Account created successfully!"}


def test_register_duplicate_email():
    """Test: Register with same email twice should fail"""
    email = unique_email()
    # Register first time
    client.post("/register", json={
        "email": email,
        "password": "test1234"
    })
    # Register second time with same email
    response = client.post("/register", json={
        "email": email,
        "password": "test1234"
    })
    # Should return 400 Bad Request
    assert response.status_code == 400


def test_login_success():
    """Test: Login with correct credentials"""
    email = unique_email()
    # First register
    client.post("/register", json={
        "email": email,
        "password": "test1234"
    })
    # Then login
    response = client.post("/login", data={
        "username": email,
        "password": "test1234"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()  # must have token


def test_login_wrong_password():
    """Test: Login with wrong password should fail"""
    email = unique_email()
    client.post("/register", json={
        "email": email,
        "password": "test1234"
    })
    response = client.post("/login", data={
        "username": email,
        "password": "wrongpassword"
    })
    assert response.status_code == 401


# -----------------------------------------------
# EMPLOYEE TESTS
# -----------------------------------------------

def get_token():
    """Helper: Register and login to get a fresh token"""
    email = unique_email()
    client.post("/register", json={
        "email": email,
        "password": "test1234"
    })
    response = client.post("/login", data={
        "username": email,
        "password": "test1234"
    })
    return response.json()["access_token"]


def test_create_employee():
    """Test: Create a new employee"""
    token = get_token()
    response = client.post("/employees",
        json={
            "name": "John Doe",
            "email": unique_email(),
            "department": "Engineering",
            "role": "Developer",
            "salary": 75000,
            "join_date": "2024-01-01"
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "John Doe"


def test_get_all_employees():
    """Test: Get all employees"""
    token = get_token()
    response = client.get("/employees",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)  # must return a list


def test_get_employees_without_token():
    """Test: Access employees without token should fail"""
    response = client.get("/employees")
    assert response.status_code == 401  # must be unauthorized


def test_create_employee_invalid_data():
    """Test: Create employee with bad data should fail"""
    token = get_token()
    response = client.post("/employees",
        json={
            "name": "J",           # too short! min 2 chars
            "email": "not-an-email",  # invalid email!
            "department": "Eng",
            "role": "Dev",
            "salary": -100,        # negative salary!
            "join_date": "2024-01-01"
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 422  # validation error
