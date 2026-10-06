from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class UploadUpdate(BaseModel):
    """Payload for PATCH /uploads/{id} — only mutable fields are included."""

    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    is_selected: bool | None = Field(None, description="Whether this upload is selected")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    id: str
    conversation_id: str
    name: str
    file_type: str
    is_selected: bool
    storage_path: str

    model_config = ConfigDict(from_attributes=True)
