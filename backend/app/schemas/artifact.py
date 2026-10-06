from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ArtifactUpdate(BaseModel):
    """Payload for PATCH /artifacts/{id} — only mutable fields are included."""

    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ArtifactResponse(BaseModel):
    id: str
    conversation_id: str | None
    source_upload_id: str | None
    name: str
    file_type: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
