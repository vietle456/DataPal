import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.conversation import Conversation
from app.models.user import User
from app.schemas.conversation import ConversationCreate, ConversationResponse, ConversationUpdate

router = APIRouter(prefix="/conversations", tags=["Conversations"])


# ---------------------------------------------------------------------------
# GET /conversations/
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=list[ConversationResponse],
    summary="List all conversations",
    description="Return all conversations that belong to the authenticated user.",
)
async def list_conversations(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ConversationResponse]:
    result = await db.scalars(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    return [ConversationResponse.model_validate(c) for c in result.all()]


# ---------------------------------------------------------------------------
# GET /conversations/{conversation_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
    summary="Get a conversation by ID",
)
async def get_conversation(
    conversation_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ConversationResponse:
    conversation = await _get_owned_conversation(conversation_id, current_user, db)
    return ConversationResponse.model_validate(conversation)


# ---------------------------------------------------------------------------
# POST /conversations/
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
)
async def create_conversation(
    body: ConversationCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ConversationResponse:
    new_id = str(uuid.uuid4())
    conversation = Conversation(
        id=new_id,
        user_id=current_user.id,
        title=body.title,
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return ConversationResponse.model_validate(conversation)


# ---------------------------------------------------------------------------
# PATCH /conversations/{conversation_id}
# ---------------------------------------------------------------------------


@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
    summary="Update a conversation",
)
async def update_conversation(
    conversation_id: str,
    body: ConversationUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ConversationResponse:
    conversation = await _get_owned_conversation(conversation_id, current_user, db)
    conversation.title = body.title
    await db.commit()
    await db.refresh(conversation)
    return ConversationResponse.model_validate(conversation)


# ---------------------------------------------------------------------------
# DELETE /conversations/{conversation_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation",
)
async def delete_conversation(
    conversation_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    conversation = await _get_owned_conversation(conversation_id, current_user, db)
    await db.delete(conversation)
    await db.commit()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


async def _get_owned_conversation(
    conversation_id: str,
    current_user: User,
    db: AsyncSession,
) -> Conversation:
    """Fetch a :class:`Conversation` and verify it belongs to *current_user*."""
    conversation: Conversation | None = await db.scalar(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    if conversation.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")
    return conversation
