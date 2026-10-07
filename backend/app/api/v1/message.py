import asyncio
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import stream_graph
from app.core.database import AsyncSessionLocal, get_db
from app.core.deps import get_current_user
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User
from app.schemas.message import MessageCreate, MessageResponse

router = APIRouter(prefix="/conversations", tags=["Messages"])


async def _persist_assistant_reply(conversation_id: str, content: str) -> None:
    """Save the assistant reply using its own short-lived session."""
    async with AsyncSessionLocal() as session:
        session.add(
            Message(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role="assistant",
                content=content,
            )
        )
        await session.commit()


# ---------------------------------------------------------------------------
# GET /conversations/{conversation_id}/messages/
# ---------------------------------------------------------------------------


@router.get(
    "/{conversation_id}/messages",
    response_model=list[MessageResponse],
    summary="List all messages in a conversation",
)
async def list_messages(
    conversation_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[MessageResponse]:
    await _assert_conversation_owned(conversation_id, current_user, db)
    result = await db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    return [MessageResponse.model_validate(m) for m in result.all()]


# ---------------------------------------------------------------------------
# POST /conversations/{conversation_id}/messages/
# ---------------------------------------------------------------------------


@router.post(
    "/{conversation_id}/messages",
    status_code=status.HTTP_201_CREATED,
    response_model=None,
    summary="Create a new message in a conversation",
)
async def create_message(
    conversation_id: str,
    body: MessageCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StreamingResponse | MessageResponse:
    await _assert_conversation_owned(conversation_id, current_user, db)

    # ── 1. Persist the incoming message ──────────────────────────────────────
    user_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=body.role,
        content=body.content,
    )
    db.add(user_msg)
    await db.commit()
    await db.refresh(user_msg)

    # ── 2. Non-user turns: just return the saved message ─────────────────────
    if body.role != "user":
        return MessageResponse.model_validate(user_msg)

    # ── 3. User turns: stream agent progress via SSE ─────────────────────────
    async def _event_stream():
        """Yield SSE lines from the agent; persist the assistant reply at the end."""
        final_answer = ""
        import json

        try:
            async for chunk in stream_graph(body.content, conversation_id):
                # Intercept the `done` sentinel to capture the answer
                if chunk.startswith("data: "):
                    try:
                        payload = json.loads(chunk[len("data: ") :].strip())
                        if payload.get("type") == "done":
                            final_answer = payload.get("content", "")
                    except (json.JSONDecodeError, ValueError):
                        pass
                yield chunk
        except Exception as exc:  # noqa: BLE001
            error_payload = json.dumps({"type": "error", "content": str(exc)})
            yield f"data: {error_payload}\n\n"
            final_answer = "I encountered an error and could not complete your request."
        finally:
            # Always persist the assistant reply so the conversation is consistent.
            # Use an independent session (the request-scoped `db` may already be
            # closed) and shield the write so a client disconnect (task
            # cancellation) cannot abort the commit midway.
            if not final_answer:
                final_answer = "I was unable to process your request."
            await asyncio.shield(_persist_assistant_reply(conversation_id, final_answer))

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable Nginx buffering if behind a proxy
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _assert_conversation_owned(
    conversation_id: str,
    current_user: User,
    db: AsyncSession,
) -> None:
    """Raise 404/403 if the conversation doesn't exist or isn't owned by *current_user*."""
    conversation: Conversation | None = await db.scalar(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    if conversation.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")
