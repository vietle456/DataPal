"""Tests for GET/POST/PATCH/DELETE /api/v1/conversations/{id}/messages.

Note: POST (create_message) with role='user' triggers the AI agent via SSE.
      We mock `stream_graph` to avoid real LLM calls.
"""

import json
from unittest.mock import patch

import pytest
from httpx import AsyncClient

from tests.conftest import auth_headers, create_conversation, create_message, create_user

BASE = "/api/v1/conversations"


# ---------------------------------------------------------------------------
# GET /conversations/{id}/messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_messages_empty(client: AsyncClient, db_session):
    user = await create_user(db_session, username="listmsg1")
    conv = await create_conversation(db_session, user.id)
    resp = await client.get(f"{BASE}/{conv.id}/messages", headers=auth_headers(user.id))
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_messages_returns_all(client: AsyncClient, db_session):
    user = await create_user(db_session, username="listmsg2")
    conv = await create_conversation(db_session, user.id)
    await create_message(db_session, conv.id, role="user", content="Hi")
    await create_message(db_session, conv.id, role="assistant", content="Hello!")

    resp = await client.get(f"{BASE}/{conv.id}/messages", headers=auth_headers(user.id))
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_list_messages_conversation_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="listmsg3")
    resp = await client.get(f"{BASE}/ghost-conv/messages", headers=auth_headers(user.id))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_messages_forbidden(client: AsyncClient, db_session):
    owner = await create_user(db_session, username="msgowner1")
    intruder = await create_user(db_session, username="msgintruder1")
    conv = await create_conversation(db_session, owner.id)
    resp = await client.get(f"{BASE}/{conv.id}/messages", headers=auth_headers(intruder.id))
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# GET /conversations/{id}/messages/{message_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_message_success(client: AsyncClient, db_session):
    user = await create_user(db_session, username="getmsg1")
    conv = await create_conversation(db_session, user.id)
    msg = await create_message(db_session, conv.id, content="Find me")

    resp = await client.get(f"{BASE}/{conv.id}/messages/{msg.id}", headers=auth_headers(user.id))
    assert resp.status_code == 200
    assert resp.json()["content"] == "Find me"


@pytest.mark.asyncio
async def test_get_message_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="getmsg2")
    conv = await create_conversation(db_session, user.id)
    resp = await client.get(f"{BASE}/{conv.id}/messages/no-such-msg", headers=auth_headers(user.id))
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /conversations/{id}/messages  (non-user role — no agent)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_message_assistant_role(client: AsyncClient, db_session):
    """Non-user messages are saved directly without triggering the agent."""
    user = await create_user(db_session, username="createmsg1")
    conv = await create_conversation(db_session, user.id)
    payload = {"role": "assistant", "content": "I am the bot"}
    resp = await client.post(
        f"{BASE}/{conv.id}/messages", json=payload, headers=auth_headers(user.id)
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["content"] == "I am the bot"
    assert data["role"] == "assistant"


@pytest.mark.asyncio
async def test_create_message_user_role_streams(client: AsyncClient, db_session):
    """User messages trigger SSE streaming; we mock stream_graph."""
    user = await create_user(db_session, username="createmsg2")
    conv = await create_conversation(db_session, user.id)

    done_event = json.dumps({"type": "done", "content": "42 is the answer"})

    async def fake_stream(_content, _conversation_id):
        yield f"data: {done_event}\n\n"

    with patch("app.api.v1.message.stream_graph", new=fake_stream):
        resp = await client.post(
            f"{BASE}/{conv.id}/messages",
            json={"role": "user", "content": "What is the answer?"},
            headers=auth_headers(user.id),
        )
    assert resp.status_code == 200  # StreamingResponse ignores status_code=201
    # SSE body should contain the done event
    assert done_event in resp.text


async def _start_stream(db_session, user, conv, fake_stream, monkeypatch):
    """Call the route handler directly and return the StreamingResponse.

    stream_graph stays patched until the test ends: the SSE generator runs
    lazily, after this handler has already returned.
    """
    from app.api.v1.message import create_message as create_message_route
    from app.schemas.message import MessageCreate

    monkeypatch.setattr("app.api.v1.message.stream_graph", fake_stream)
    return await create_message_route(
        conv.id, MessageCreate(role="user", content="hi"), user, db_session
    )


async def _assistant_messages(conv_id):
    from sqlalchemy import select

    from app.api.v1 import message as message_module
    from app.models.message import Message

    async with message_module.AsyncSessionLocal() as s:
        result = await s.execute(
            select(Message).where(Message.conversation_id == conv_id, Message.role == "assistant")
        )
        return list(result.scalars())


@pytest.mark.asyncio
async def test_stream_client_disconnect_closes_generator_and_persists(db_session, monkeypatch):
    """Early disconnect (generator closed mid-stream) still saves the reply."""
    user = await create_user(db_session, username="disconnect1")
    conv = await create_conversation(db_session, user.id)

    async def fake_stream(_c, _id):
        yield 'data: {"type": "progress"}\n\n'
        yield 'data: {"type": "done", "content": "never sent"}\n\n'

    resp = await _start_stream(db_session, user, conv, fake_stream, monkeypatch)
    body = resp.body_iterator
    await body.__anext__()  # client reads one chunk...
    await body.aclose()  # ...then disconnects

    msgs = await _assistant_messages(conv.id)
    assert len(msgs) == 1
    assert msgs[0].content == "I was unable to process your request."


@pytest.mark.asyncio
async def test_stream_cancellation_does_not_abort_persist(db_session, monkeypatch):
    """Task cancellation (Starlette on disconnect) must not abort the commit."""
    import asyncio

    user = await create_user(db_session, username="disconnect2")
    conv = await create_conversation(db_session, user.id)

    async def fake_stream(_c, _id):
        yield 'data: {"type": "progress"}\n\n'
        await asyncio.sleep(60)

    resp = await _start_stream(db_session, user, conv, fake_stream, monkeypatch)

    async def consume():
        async for _ in resp.body_iterator:
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.2)  # let the shielded write finish

    assert len(await _assistant_messages(conv.id)) == 1


@pytest.mark.asyncio
async def test_create_message_conv_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="createmsg3")
    resp = await client.post(
        f"{BASE}/ghost/messages",
        json={"role": "user", "content": "Hi"},
        headers=auth_headers(user.id),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_message_forbidden(client: AsyncClient, db_session):
    owner = await create_user(db_session, username="msgowner2")
    intruder = await create_user(db_session, username="msgintruder2")
    conv = await create_conversation(db_session, owner.id)
    resp = await client.post(
        f"{BASE}/{conv.id}/messages",
        json={"role": "assistant", "content": "Hi"},
        headers=auth_headers(intruder.id),
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# PATCH /conversations/{id}/messages/{message_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_message_success(client: AsyncClient, db_session):
    user = await create_user(db_session, username="updatemsg1")
    conv = await create_conversation(db_session, user.id)
    msg = await create_message(db_session, conv.id, content="Old content")

    resp = await client.patch(
        f"{BASE}/{conv.id}/messages/{msg.id}",
        json={"content": "Updated content"},
        headers=auth_headers(user.id),
    )
    assert resp.status_code == 200
    assert resp.json()["content"] == "Updated content"


@pytest.mark.asyncio
async def test_update_message_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="updatemsg2")
    conv = await create_conversation(db_session, user.id)
    resp = await client.patch(
        f"{BASE}/{conv.id}/messages/ghost",
        json={"content": "nope"},
        headers=auth_headers(user.id),
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /conversations/{id}/messages/{message_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_message_success(client: AsyncClient, db_session):
    user = await create_user(db_session, username="delmsg1")
    conv = await create_conversation(db_session, user.id)
    msg = await create_message(db_session, conv.id)
    resp = await client.delete(f"{BASE}/{conv.id}/messages/{msg.id}", headers=auth_headers(user.id))
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_delete_message_not_found(client: AsyncClient, db_session):
    user = await create_user(db_session, username="delmsg2")
    conv = await create_conversation(db_session, user.id)
    resp = await client.delete(f"{BASE}/{conv.id}/messages/no-such", headers=auth_headers(user.id))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_message_forbidden(client: AsyncClient, db_session):
    owner = await create_user(db_session, username="msgowner3")
    intruder = await create_user(db_session, username="msgintruder3")
    conv = await create_conversation(db_session, owner.id)
    msg = await create_message(db_session, conv.id)
    resp = await client.delete(
        f"{BASE}/{conv.id}/messages/{msg.id}", headers=auth_headers(intruder.id)
    )
    assert resp.status_code == 403
