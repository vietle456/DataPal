from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.conversation import Conversation
    from app.models.upload import Upload


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    conversation_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("conversations.id"), nullable=True
    )
    source_upload_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("uploads.id"), nullable=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    file_type: Mapped[str] = mapped_column(String, nullable=False)

    # Relationships
    conversation: Mapped[Optional["Conversation"]] = relationship(
        "Conversation", back_populates="artifacts"
    )
    source_upload: Mapped[Optional["Upload"]] = relationship("Upload", back_populates="artifacts")
