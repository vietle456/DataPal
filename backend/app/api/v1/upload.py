import os
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_conversation_storage_paths
from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.conversation import Conversation
from app.models.upload import Upload
from app.models.user import User
from app.schemas.upload import UploadResponse, UploadUpdate

router = APIRouter(prefix="/conversations", tags=["Uploads"])


# ---------------------------------------------------------------------------
# GET /conversations/{conversation_id}/uploads
# ---------------------------------------------------------------------------


@router.get(
    "/{conversation_id}/uploads",
    response_model=list[UploadResponse],
    summary="List all uploads in a conversation",
)
async def list_uploads(
    conversation_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[UploadResponse]:
    await _assert_conversation_owned(conversation_id, current_user, db)
    result = await db.scalars(select(Upload).where(Upload.conversation_id == conversation_id))
    return [UploadResponse.model_validate(u) for u in result.all()]


# ---------------------------------------------------------------------------
# GET /conversations/{conversation_id}/uploads/{upload_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{conversation_id}/uploads/{upload_id}",
    response_model=UploadResponse,
    summary="Get an upload by ID",
)
async def get_upload(
    conversation_id: str,
    upload_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UploadResponse:
    await _assert_conversation_owned(conversation_id, current_user, db)
    upload = await _get_upload(upload_id, conversation_id, db)
    return UploadResponse.model_validate(upload)


# ---------------------------------------------------------------------------
# POST /conversations/{conversation_id}/uploads
# ---------------------------------------------------------------------------


@router.post(
    "/{conversation_id}/uploads",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file to a conversation",
)
async def create_upload(
    conversation_id: str,
    file: Annotated[UploadFile, File(description="The file to upload")],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UploadResponse:
    await _assert_conversation_owned(conversation_id, current_user, db)

    upload_id = str(uuid.uuid4())
    filename = file.filename or upload_id
    file_type = file.content_type or "application/octet-stream"

    # Persist file to per-conversation local storage
    paths = get_conversation_storage_paths(conversation_id)
    uploads_dir = paths["uploads"]
    uploads_dir.mkdir(parents=True, exist_ok=True)
    storage_path = str(uploads_dir / f"{upload_id}_{filename}")

    contents = await file.read()
    with open(storage_path, "wb") as f:
        f.write(contents)

    upload = Upload(
        id=upload_id,
        conversation_id=conversation_id,
        name=filename,
        file_type=file_type,
        storage_path=storage_path,
        is_selected=False,
    )
    db.add(upload)
    await db.commit()
    await db.refresh(upload)

    # Ingest CSV or Parquet into conversation-scoped DuckDB database
    lower_name = filename.lower()
    if lower_name.endswith(".csv") or lower_name.endswith(".parquet"):
        import re
        from pathlib import Path

        from app.services.duckdb_engine import DuckDBEngine

        db_path = paths["db"]
        db_path.parent.mkdir(parents=True, exist_ok=True)

        raw_stem = Path(filename).stem
        sanitized_table = re.sub(r"[^a-zA-Z0-9_]", "_", raw_stem).lower()
        if not sanitized_table or sanitized_table[0].isdigit():
            sanitized_table = f"t_{sanitized_table}"

        # Connect with write access, load dataset, and close immediately
        with DuckDBEngine(db_path, read_only=False) as duck_engine:
            duck_engine.load_dataset(file_path=Path(storage_path), table_name=sanitized_table)

    return UploadResponse.model_validate(upload)


# ---------------------------------------------------------------------------
# PATCH /conversations/{conversation_id}/uploads/{upload_id}
# ---------------------------------------------------------------------------


@router.patch(
    "/{conversation_id}/uploads/{upload_id}",
    response_model=UploadResponse,
    summary="Update upload metadata",
)
async def update_upload(
    conversation_id: str,
    upload_id: str,
    body: UploadUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UploadResponse:
    await _assert_conversation_owned(conversation_id, current_user, db)
    upload = await _get_upload(upload_id, conversation_id, db)

    if body.name is not None:
        upload.name = body.name
    if body.is_selected is not None:
        upload.is_selected = body.is_selected

    await db.commit()
    await db.refresh(upload)
    return UploadResponse.model_validate(upload)


# ---------------------------------------------------------------------------
# DELETE /conversations/{conversation_id}/uploads/{upload_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/{conversation_id}/uploads/{upload_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an upload",
)
async def delete_upload(
    conversation_id: str,
    upload_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    await _assert_conversation_owned(conversation_id, current_user, db)
    upload = await _get_upload(upload_id, conversation_id, db)

    # Remove file from disk if it exists
    if os.path.isfile(upload.storage_path):
        os.remove(upload.storage_path)

    await db.delete(upload)
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


async def _get_upload(upload_id: str, conversation_id: str, db: AsyncSession) -> Upload:
    upload: Upload | None = await db.scalar(
        select(Upload).where(
            Upload.id == upload_id,
            Upload.conversation_id == conversation_id,
        )
    )
    if upload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found.")
    return upload
