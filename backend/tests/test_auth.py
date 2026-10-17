import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db.session import Base, engine, SessionLocal
from app.models.user import User

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    # Cleanup created users after test
    db = SessionLocal()
    db.query(User).filter(User.email.like("%test%@example.com")).delete()
    db.commit()
    db.close()


def test_user_registration_and_login():
    email = "testuser1@example.com"
    password = "password123"

    # 1. Register
    reg_resp = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert reg_resp.status_code == 201
    reg_data = reg_resp.json()
    assert reg_data["email"] == email
    assert "id" in reg_data

    # 2. Duplicate registration fails
    dup_resp = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert dup_resp.status_code == 400

    # 3. Login with wrong password fails
    bad_login = client.post("/api/v1/auth/login", json={"email": email, "password": "wrongpassword"})
    assert bad_login.status_code == 401

    # 4. Login with correct password succeeds
    good_login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert good_login.status_code == 200
    token_data = good_login.json()
    assert "access_token" in token_data
    token = token_data["access_token"]

    # 5. Access /me profile
    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == email


def test_unauthenticated_requests_blocked():
    # Calling /upload or /query without token should be blocked with 401
    query_resp = client.post("/api/v1/query", json={"question": "hello"})
    assert query_resp.status_code == 401
    assert "Authentication token required" in query_resp.json()["detail"]


def test_invalid_token_rejected():
    query_resp = client.post(
        "/api/v1/query",
        json={"question": "hello"},
        headers={"Authorization": "Bearer invalid.token.value"}
    )
    assert query_resp.status_code == 401
