from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ConversationCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Conversation title")


class ConversationUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="New conversation title")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    title: str
    last_message_id: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
