"""Tests for GET/POST/PATCH/DELETE /api/v1/conversations/{id}/uploads."""

import io

import pytest
from httpx import AsyncClient

from tests.conftest import auth_headers, create_conversation, create_upload, create_user

BASE = "/api/v1/conversations"


# ---------------------------------------------------------------------------
# GET /conversations/{id}/uploads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_uploads_empty(client: AsyncClient, db_session):
    user = await create_user(db_session, username="listup1")
    conv = await create_conversation(db_session, user.id)
    resp = await client.get(f"{BASE}/{conv.id}/uploads", headers=auth_headers(user.id))
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_uploads_returns_all(client: AsyncClient, db_session):
    user = await create_user(db_session, username="listup2")
    conv = await create_conversation(db_session, user.id)
    await create_upload(db_session, conv.id, name="a.csv")
    await create_upload(db_session, conv.id, name="b.csv")

    resp = await client.get(f"{BASE}/{conv.id}/uploads", headers=auth_headers(user.id))
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_list_uploads_conv_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="listup3")
    resp = await client.get(f"{BASE}/ghost/uploads", headers=auth_headers(user.id))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_uploads_forbidden(client: AsyncClient, db_session):
    owner = await create_user(db_session, username="upowner1")
    intruder = await create_user(db_session, username="upintruder1")
    conv = await create_conversation(db_session, owner.id)
    resp = await client.get(f"{BASE}/{conv.id}/uploads", headers=auth_headers(intruder.id))
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# GET /conversations/{id}/uploads/{upload_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_upload_success(client: AsyncClient, db_session):
    user = await create_user(db_session, username="getup1")
    conv = await create_conversation(db_session, user.id)
    upload = await create_upload(db_session, conv.id, name="data.csv")

    resp = await client.get(f"{BASE}/{conv.id}/uploads/{upload.id}", headers=auth_headers(user.id))
    assert resp.status_code == 200
    assert resp.json()["name"] == "data.csv"


@pytest.mark.asyncio
async def test_get_upload_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="getup2")
    conv = await create_conversation(db_session, user.id)
    resp = await client.get(f"{BASE}/{conv.id}/uploads/ghost-id", headers=auth_headers(user.id))
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /conversations/{id}/uploads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_upload_success(client: AsyncClient, db_session, tmp_path):
    user = await create_user(db_session, username="createup1")
    conv = await create_conversation(db_session, user.id)

    file_content = b"col1,col2\n1,2\n3,4"
    # Redirect file storage to pytest's tmp_path so nothing lands in storage/
    from unittest.mock import patch

    def fake_paths(conversation_id: str) -> dict:
        base = tmp_path / conversation_id
        return {
            "db": base / "db.duckdb",
            "uploads": base / "input" / "uploads",
            "sql_results": base / "intermediate" / "sql_results",
            "artifacts": base / "output" / "artifacts",
        }

    with patch("app.api.v1.upload.get_conversation_storage_paths", side_effect=fake_paths):
        resp = await client.post(
            f"{BASE}/{conv.id}/uploads",
            files={"file": ("test.csv", io.BytesIO(file_content), "text/csv")},
            headers=auth_headers(user.id),
        )

    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "test.csv"
    assert data["file_type"] == "text/csv"
    assert data["conversation_id"] == conv.id


@pytest.mark.asyncio
async def test_create_upload_conv_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="createup2")
    resp = await client.post(
        f"{BASE}/no-conv/uploads",
        files={"file": ("f.csv", io.BytesIO(b"data"), "text/csv")},
        headers=auth_headers(user.id),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_upload_forbidden(client: AsyncClient, db_session):
    owner = await create_user(db_session, username="upowner2")
    intruder = await create_user(db_session, username="upintruder2")
    conv = await create_conversation(db_session, owner.id)
    resp = await client.post(
        f"{BASE}/{conv.id}/uploads",
        files={"file": ("f.csv", io.BytesIO(b"data"), "text/csv")},
        headers=auth_headers(intruder.id),
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# PATCH /conversations/{id}/uploads/{upload_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_upload_name(client: AsyncClient, db_session):
    user = await create_user(db_session, username="updateup1")
    conv = await create_conversation(db_session, user.id)
    upload = await create_upload(db_session, conv.id, name="old.csv")

    resp = await client.patch(
        f"{BASE}/{conv.id}/uploads/{upload.id}",
        json={"name": "new.csv"},
        headers=auth_headers(user.id),
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "new.csv"


@pytest.mark.asyncio
async def test_update_upload_is_selected(client: AsyncClient, db_session):
    user = await create_user(db_session, username="updateup2")
    conv = await create_conversation(db_session, user.id)
    upload = await create_upload(db_session, conv.id)

    resp = await client.patch(
        f"{BASE}/{conv.id}/uploads/{upload.id}",
        json={"is_selected": True},
        headers=auth_headers(user.id),
    )
    assert resp.status_code == 200
    assert resp.json()["is_selected"] is True


@pytest.mark.asyncio
async def test_update_upload_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="updateup3")
    conv = await create_conversation(db_session, user.id)
    resp = await client.patch(
        f"{BASE}/{conv.id}/uploads/ghost",
        json={"name": "nope"},
        headers=auth_headers(user.id),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_upload_forbidden(client: AsyncClient, db_session):
    owner = await create_user(db_session, username="upowner3")
    intruder = await create_user(db_session, username="upintruder3")
    conv = await create_conversation(db_session, owner.id)
    upload = await create_upload(db_session, conv.id)
    resp = await client.patch(
        f"{BASE}/{conv.id}/uploads/{upload.id}",
        json={"name": "hack"},
        headers=auth_headers(intruder.id),
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# DELETE /conversations/{id}/uploads/{upload_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_upload_success(client: AsyncClient, db_session):
    user = await create_user(db_session, username="delup1")
    conv = await create_conversation(db_session, user.id)
    # storage_path points to a non-existent file — delete_upload handles that gracefully
    upload = await create_upload(db_session, conv.id, storage_path="/nonexistent/path.csv")
    resp = await client.delete(
        f"{BASE}/{conv.id}/uploads/{upload.id}", headers=auth_headers(user.id)
    )
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_delete_upload_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="delup2")
    conv = await create_conversation(db_session, user.id)
    resp = await client.delete(f"{BASE}/{conv.id}/uploads/ghost", headers=auth_headers(user.id))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_upload_forbidden(client: AsyncClient, db_session):
    owner = await create_user(db_session, username="upowner4")
    intruder = await create_user(db_session, username="upintruder4")
    conv = await create_conversation(db_session, owner.id)
    upload = await create_upload(db_session, conv.id)
    resp = await client.delete(
        f"{BASE}/{conv.id}/uploads/{upload.id}", headers=auth_headers(intruder.id)
    )
    assert resp.status_code == 403
