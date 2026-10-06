from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.artifact import Artifact
from app.models.conversation import Conversation
from app.models.user import User
from app.schemas.artifact import ArtifactResponse, ArtifactUpdate

router = APIRouter(prefix="/conversations", tags=["Artifacts"])


# ---------------------------------------------------------------------------
# GET /conversations/{conversation_id}/artifacts
# ---------------------------------------------------------------------------


@router.get(
    "/{conversation_id}/artifacts",
    response_model=list[ArtifactResponse],
    summary="List all artifacts in a conversation",
)
async def list_artifacts(
    conversation_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ArtifactResponse]:
    await _assert_conversation_owned(conversation_id, current_user, db)
    result = await db.scalars(
        select(Artifact)
        .where(Artifact.conversation_id == conversation_id)
        .order_by(Artifact.created_at.asc())
    )
    return [ArtifactResponse.model_validate(a) for a in result.all()]


# ---------------------------------------------------------------------------
# GET /conversations/{conversation_id}/artifacts/{artifact_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{conversation_id}/artifacts/{artifact_id}",
    response_model=ArtifactResponse,
    summary="Get an artifact by ID",
)
async def get_artifact(
    conversation_id: str,
    artifact_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ArtifactResponse:
    await _assert_conversation_owned(conversation_id, current_user, db)
    artifact = await _get_artifact(artifact_id, conversation_id, db)
    return ArtifactResponse.model_validate(artifact)


# ---------------------------------------------------------------------------
# PATCH /conversations/{conversation_id}/artifacts/{artifact_id}
# ---------------------------------------------------------------------------


@router.patch(
    "/{conversation_id}/artifacts/{artifact_id}",
    response_model=ArtifactResponse,
    summary="Update artifact metadata",
)
async def update_artifact(
    conversation_id: str,
    artifact_id: str,
    body: ArtifactUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ArtifactResponse:
    await _assert_conversation_owned(conversation_id, current_user, db)
    artifact = await _get_artifact(artifact_id, conversation_id, db)

    if body.name is not None:
        artifact.name = body.name

    await db.commit()
    await db.refresh(artifact)
    return ArtifactResponse.model_validate(artifact)


# ---------------------------------------------------------------------------
# DELETE /conversations/{conversation_id}/artifacts/{artifact_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/{conversation_id}/artifacts/{artifact_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an artifact",
)
async def delete_artifact(
    conversation_id: str,
    artifact_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _assert_conversation_owned(conversation_id, current_user, db)
    artifact = await _get_artifact(artifact_id, conversation_id, db)
    await db.delete(artifact)
    await db.commit()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _assert_conversation_owned(
    conversation_id: str,
    current_user: User,
    db: AsyncSession,
) -> None:
    conversation: Conversation | None = await db.scalar(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    if conversation.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")


async def _get_artifact(artifact_id: str, conversation_id: str, db: AsyncSession) -> Artifact:
    artifact: Artifact | None = await db.scalar(
        select(Artifact).where(
            Artifact.id == artifact_id,
            Artifact.conversation_id == conversation_id,
        )
    )
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found.")
    return artifact
