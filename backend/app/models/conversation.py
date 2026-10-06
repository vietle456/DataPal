from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.artifact import Artifact
    from app.models.message import Message
    from app.models.upload import Upload
    from app.models.user import User


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    last_message_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("messages.id", use_alter=True, name="fk_conversation_last_message"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    title: Mapped[str] = mapped_column(String, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="conversations")
    last_message: Mapped[Optional["Message"]] = relationship(
        "Message", foreign_keys=[last_message_id], post_update=True
    )
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        foreign_keys="Message.conversation_id",
        cascade="all, delete-orphan",
    )
    uploads: Mapped[list["Upload"]] = relationship(
        "Upload", back_populates="conversation", cascade="all, delete-orphan"
    )
    artifacts: Mapped[list["Artifact"]] = relationship(
        "Artifact", back_populates="conversation", cascade="all, delete-orphan"
    )
