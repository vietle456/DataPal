"""Tests for GET/PATCH/DELETE /api/v1/conversations/{id}/artifacts."""

import pytest
from httpx import AsyncClient

from tests.conftest import auth_headers, create_artifact, create_conversation, create_user

BASE = "/api/v1/conversations"


# ---------------------------------------------------------------------------
# GET /conversations/{id}/artifacts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_artifacts_empty(client: AsyncClient, db_session):
    user = await create_user(db_session, username="listart1")
    conv = await create_conversation(db_session, user.id)
    resp = await client.get(f"{BASE}/{conv.id}/artifacts", headers=auth_headers(user.id))
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_artifacts_returns_all(client: AsyncClient, db_session):
    user = await create_user(db_session, username="listart2")
    conv = await create_conversation(db_session, user.id)
    await create_artifact(db_session, conv.id, name="a.json")
    await create_artifact(db_session, conv.id, name="b.csv")

    resp = await client.get(f"{BASE}/{conv.id}/artifacts", headers=auth_headers(user.id))
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_list_artifacts_conv_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="listart3")
    resp = await client.get(f"{BASE}/ghost/artifacts", headers=auth_headers(user.id))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_artifacts_forbidden(client: AsyncClient, db_session):
    owner = await create_user(db_session, username="artowner1")
    intruder = await create_user(db_session, username="artintruder1")
    conv = await create_conversation(db_session, owner.id)
    resp = await client.get(f"{BASE}/{conv.id}/artifacts", headers=auth_headers(intruder.id))
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# GET /conversations/{id}/artifacts/{artifact_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_artifact_success(client: AsyncClient, db_session):
    user = await create_user(db_session, username="getart1")
    conv = await create_conversation(db_session, user.id)
    artifact = await create_artifact(db_session, conv.id, name="result.json")

    resp = await client.get(
        f"{BASE}/{conv.id}/artifacts/{artifact.id}", headers=auth_headers(user.id)
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "result.json"
    assert data["conversation_id"] == conv.id


@pytest.mark.asyncio
async def test_get_artifact_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="getart2")
    conv = await create_conversation(db_session, user.id)
    resp = await client.get(f"{BASE}/{conv.id}/artifacts/ghost-id", headers=auth_headers(user.id))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_artifact_forbidden(client: AsyncClient, db_session):
    owner = await create_user(db_session, username="artowner2")
    intruder = await create_user(db_session, username="artintruder2")
    conv = await create_conversation(db_session, owner.id)
    artifact = await create_artifact(db_session, conv.id)
    resp = await client.get(
        f"{BASE}/{conv.id}/artifacts/{artifact.id}", headers=auth_headers(intruder.id)
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# PATCH /conversations/{id}/artifacts/{artifact_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_artifact_name(client: AsyncClient, db_session):
    user = await create_user(db_session, username="updateart1")
    conv = await create_conversation(db_session, user.id)
    artifact = await create_artifact(db_session, conv.id, name="old_name.json")

    resp = await client.patch(
        f"{BASE}/{conv.id}/artifacts/{artifact.id}",
        json={"name": "new_name.json"},
        headers=auth_headers(user.id),
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "new_name.json"


@pytest.mark.asyncio
async def test_update_artifact_null_name_no_change(client: AsyncClient, db_session):
    """Sending name=null should leave the artifact name unchanged."""
    user = await create_user(db_session, username="updateart2")
    conv = await create_conversation(db_session, user.id)
    artifact = await create_artifact(db_session, conv.id, name="keep.json")

    resp = await client.patch(
        f"{BASE}/{conv.id}/artifacts/{artifact.id}",
        json={"name": None},
        headers=auth_headers(user.id),
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "keep.json"


@pytest.mark.asyncio
async def test_update_artifact_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="updateart3")
    conv = await create_conversation(db_session, user.id)
    resp = await client.patch(
        f"{BASE}/{conv.id}/artifacts/ghost",
        json={"name": "nope"},
        headers=auth_headers(user.id),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_artifact_forbidden(client: AsyncClient, db_session):
    owner = await create_user(db_session, username="artowner3")
    intruder = await create_user(db_session, username="artintruder3")
    conv = await create_conversation(db_session, owner.id)
    artifact = await create_artifact(db_session, conv.id)
    resp = await client.patch(
        f"{BASE}/{conv.id}/artifacts/{artifact.id}",
        json={"name": "hack"},
        headers=auth_headers(intruder.id),
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# DELETE /conversations/{id}/artifacts/{artifact_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_artifact_success(client: AsyncClient, db_session):
    user = await create_user(db_session, username="delart1")
    conv = await create_conversation(db_session, user.id)
    artifact = await create_artifact(db_session, conv.id)
    resp = await client.delete(
        f"{BASE}/{conv.id}/artifacts/{artifact.id}", headers=auth_headers(user.id)
    )
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_delete_artifact_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="delart2")
    conv = await create_conversation(db_session, user.id)
    resp = await client.delete(f"{BASE}/{conv.id}/artifacts/ghost", headers=auth_headers(user.id))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_artifact_forbidden(client: AsyncClient, db_session):
    owner = await create_user(db_session, username="artowner4")
    intruder = await create_user(db_session, username="artintruder4")
    conv = await create_conversation(db_session, owner.id)
    artifact = await create_artifact(db_session, conv.id)
    resp = await client.delete(
        f"{BASE}/{conv.id}/artifacts/{artifact.id}", headers=auth_headers(intruder.id)
    )
    assert resp.status_code == 403
