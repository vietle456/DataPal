import os
from contextlib import AbstractAsyncContextManager

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# The context manager instance — kept so __aexit__ closes the same pool
# that __aenter__ opened.
_checkpointer_ctx: AbstractAsyncContextManager[AsyncPostgresSaver] | None = None
checkpointer: AsyncPostgresSaver | None = None


async def init_checkpointer() -> None:
    """Create the LangGraph checkpointer pool. Call once at app startup."""
    global checkpointer, _checkpointer_ctx
    if not DATABASE_URL:
        raise ValueError("Database URL is not set")
    _checkpointer_ctx = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
    checkpointer = await _checkpointer_ctx.__aenter__()
    await checkpointer.setup()


async def close_checkpointer() -> None:
    """Tear down the checkpointer pool. Call at app shutdown."""
    global checkpointer, _checkpointer_ctx
    if _checkpointer_ctx:
        await _checkpointer_ctx.__aexit__(None, None, None)
        checkpointer = None
        _checkpointer_ctx = None
