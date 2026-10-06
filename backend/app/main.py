import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.api.v1.artifact import router as artifact_router
from app.api.v1.authentication import router as auth_router
from app.api.v1.conversation import router as conversation_router
from app.api.v1.message import router as message_router
from app.api.v1.upload import router as upload_router
from app.core.agent_checkpoint import close_checkpointer, init_checkpointer
from app.core.agent_runtime import close_agent, init_agent
from app.core.database import create_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background services, create DB tables, then serve requests."""
    await create_tables()
    await init_checkpointer()
    await init_agent()
    try:
        yield
    finally:
        await close_agent()
        await close_checkpointer()


app = FastAPI(
    lifespan=lifespan,
    title="DataPal Backend",
    version="1.0.0",
    description="REST API for the DataPal data-agent platform.",
)

API_PREFIX = "/api/v1"

# Authentication
app.include_router(auth_router, prefix=API_PREFIX)

# Conversations & nested resources
app.include_router(conversation_router, prefix=API_PREFIX)
app.include_router(message_router, prefix=API_PREFIX)
app.include_router(upload_router, prefix=API_PREFIX)
app.include_router(artifact_router, prefix=API_PREFIX)


if __name__ == "__main__":
    loop_setup = "asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto"
    uvicorn.run(app, host="localhost", port=8000, loop=loop_setup)
