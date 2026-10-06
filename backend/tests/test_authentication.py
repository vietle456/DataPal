"""Tests for POST /api/v1/auth/register and POST /api/v1/auth/login."""

import pytest
from httpx import AsyncClient

from tests.conftest import create_user

BASE = "/api/v1/auth"


# ---------------------------------------------------------------------------
# POST /auth/register
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_success(client: AsyncClient):
    payload = {"username": "newuser", "password": "securepass"}
    resp = await client.post(f"{BASE}/register", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "newuser"
    assert "user_id" in data


@pytest.mark.asyncio
async def test_register_duplicate_username(client: AsyncClient, db_session):
    await create_user(db_session, username="dupuser")
    payload = {"username": "dupuser", "password": "password123"}
    resp = await client.post(f"{BASE}/register", json=payload)
    assert resp.status_code == 409
    assert "already taken" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_register_username_too_short(client: AsyncClient):
    payload = {"username": "ab", "password": "password123"}
    resp = await client.post(f"{BASE}/register", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_register_password_too_short(client: AsyncClient):
    payload = {"username": "validuser", "password": "short"}
    resp = await client.post(f"{BASE}/register", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_register_missing_fields(client: AsyncClient):
    resp = await client.post(f"{BASE}/register", json={"username": "onlyname"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_login_success(client: AsyncClient, db_session):
    await create_user(db_session, username="loginuser", password="password123")
    payload = {"username": "loginuser", "password": "password123"}
    resp = await client.post(f"{BASE}/login", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient, db_session):
    await create_user(db_session, username="passuser", password="correctpass")
    resp = await client.post(
        f"{BASE}/login", json={"username": "passuser", "password": "wrongpass"}
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_nonexistent_user(client: AsyncClient):
    resp = await client.post(f"{BASE}/login", json={"username": "ghost", "password": "password123"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_missing_fields(client: AsyncClient):
    resp = await client.post(f"{BASE}/login", json={"username": "someone"})
    assert resp.status_code == 422
