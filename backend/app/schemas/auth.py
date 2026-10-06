from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    password: str = Field(..., min_length=8, description="Plain-text password (will be hashed)")


class LoginRequest(BaseModel):
    username: str = Field(..., description="Registered username")
    password: str = Field(..., description="Plain-text password")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    user_id: str = Field(..., validation_alias="id")
    username: str

    model_config = {"from_attributes": True, "populate_by_name": True}
