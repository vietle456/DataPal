from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class MessageCreate(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(..., description="Message author role")
    content: str = Field(..., min_length=1, description="Message text content")


class MessageUpdate(BaseModel):
    content: str = Field(..., min_length=1, description="Updated message text content")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
