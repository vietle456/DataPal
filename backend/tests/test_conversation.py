"""Tests for GET/POST/PATCH/DELETE /api/v1/conversations."""

import pytest
from httpx import AsyncClient

from tests.conftest import auth_headers, create_conversation, create_user

BASE = "/api/v1/conversations"


# ---------------------------------------------------------------------------
# GET /conversations/
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_conversations_empty(client: AsyncClient, db_session):
    user = await create_user(db_session, username="listuser1")
    resp = await client.get(f"{BASE}/", headers=auth_headers(user.id))
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_conversations_returns_own(client: AsyncClient, db_session):
    user = await create_user(db_session, username="listuser2")
    other = await create_user(db_session, username="other1")
    await create_conversation(db_session, user.id, title="Mine")
    await create_conversation(db_session, other.id, title="NotMine")

    resp = await client.get(f"{BASE}/", headers=auth_headers(user.id))
    assert resp.status_code == 200
    titles = [c["title"] for c in resp.json()]
    assert "Mine" in titles
    assert "NotMine" not in titles


@pytest.mark.asyncio
async def test_list_conversations_unauthenticated(client: AsyncClient):
    resp = await client.get(f"{BASE}/")
    assert resp.status_code in (401, 403)  # HTTPBearer: no credentials => 401/403


# ---------------------------------------------------------------------------
# GET /conversations/{conversation_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_conversation_success(client: AsyncClient, db_session):
    user = await create_user(db_session, username="getconv1")
    conv = await create_conversation(db_session, user.id, title="Fetchable")
    resp = await client.get(f"{BASE}/{conv.id}", headers=auth_headers(user.id))
    assert resp.status_code == 200
    assert resp.json()["title"] == "Fetchable"


@pytest.mark.asyncio
async def test_get_conversation_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="getconv2")
    resp = await client.get(f"{BASE}/nonexistent-id", headers=auth_headers(user.id))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_conversation_forbidden(client: AsyncClient, db_session):
    owner = await create_user(db_session, username="owner1")
    intruder = await create_user(db_session, username="intruder1")
    conv = await create_conversation(db_session, owner.id)
    resp = await client.get(f"{BASE}/{conv.id}", headers=auth_headers(intruder.id))
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# POST /conversations/
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_conversation_success(client: AsyncClient, db_session):
    user = await create_user(db_session, username="createconv1")
    resp = await client.post(f"{BASE}/", json={"title": "New Chat"}, headers=auth_headers(user.id))
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "New Chat"
    assert data["user_id"] == user.id


@pytest.mark.asyncio
async def test_create_conversation_empty_title(client: AsyncClient, db_session):
    user = await create_user(db_session, username="createconv2")
    resp = await client.post(f"{BASE}/", json={"title": ""}, headers=auth_headers(user.id))
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_conversation_unauthenticated(client: AsyncClient):
    resp = await client.post(f"{BASE}/", json={"title": "No Auth"})
    assert resp.status_code in (401, 403)  # HTTPBearer: no credentials => 401/403


# ---------------------------------------------------------------------------
# PATCH /conversations/{conversation_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_conversation_success(client: AsyncClient, db_session):
    user = await create_user(db_session, username="updateconv1")
    conv = await create_conversation(db_session, user.id, title="Old Title")
    resp = await client.patch(
        f"{BASE}/{conv.id}", json={"title": "New Title"}, headers=auth_headers(user.id)
    )
    assert resp.status_code == 200
    assert resp.json()["title"] == "New Title"


@pytest.mark.asyncio
async def test_update_conversation_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="updateconv2")
    resp = await client.patch(
        f"{BASE}/bad-id", json={"title": "Nope"}, headers=auth_headers(user.id)
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_conversation_forbidden(client: AsyncClient, db_session):
    owner = await create_user(db_session, username="owner2")
    intruder = await create_user(db_session, username="intruder2")
    conv = await create_conversation(db_session, owner.id)
    resp = await client.patch(
        f"{BASE}/{conv.id}", json={"title": "Hacked"}, headers=auth_headers(intruder.id)
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# DELETE /conversations/{conversation_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_conversation_success(client: AsyncClient, db_session):
    user = await create_user(db_session, username="delconv1")
    conv = await create_conversation(db_session, user.id)
    resp = await client.delete(f"{BASE}/{conv.id}", headers=auth_headers(user.id))
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_delete_conversation_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="delconv2")
    resp = await client.delete(f"{BASE}/nonexistent", headers=auth_headers(user.id))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_conversation_forbidden(client: AsyncClient, db_session):
    owner = await create_user(db_session, username="owner3")
    intruder = await create_user(db_session, username="intruder3")
    conv = await create_conversation(db_session, owner.id)
    resp = await client.delete(f"{BASE}/{conv.id}", headers=auth_headers(intruder.id))
    assert resp.status_code == 403
