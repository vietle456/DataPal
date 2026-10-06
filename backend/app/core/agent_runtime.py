from contextlib import AbstractAsyncContextManager

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from app.core.config import MCP_SERVER_PARAMS
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# ── Module-level singletons ────────────────────────────────────────────────────

# The compiled graph — built once at startup, reused for every request.
compiled_graph: CompiledStateGraph | None = None

# Context manager handles kept open so the MCP subprocess stays alive.
_stdio_ctx: AbstractAsyncContextManager | None = None
_session_ctx: AbstractAsyncContextManager | None = None


async def init_agent() -> None:
    """
    Start the MCP subprocess, load its tools, build and compile the LangGraph
    state graph.  Called once during application startup.
    """
    global compiled_graph, _stdio_ctx, _session_ctx

    logger.info("[init_agent] Starting long-lived MCP subprocess")
    _stdio_ctx = stdio_client(MCP_SERVER_PARAMS)
    read, write = await _stdio_ctx.__aenter__()

    _session_ctx = ClientSession(read, write)
    session = await _session_ctx.__aenter__()
    await session.initialize()

    mcp_tools = await load_mcp_tools(session)
    logger.info(
        "[init_agent] Loaded %d MCP tools: %s",
        len(mcp_tools),
        [getattr(t, "name", str(t)) for t in mcp_tools],
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Import here to avoid circular imports (graph.py imports from this module)
    from app.agent.graph import build_state_graph  # noqa: PLC0415

    compiled_graph = build_state_graph(llm, mcp_tools)
    logger.info("[init_agent] Graph compiled and ready")


async def close_agent() -> None:
    """
    Tear down the MCP session and subprocess.  Called once during application
    shutdown.
    """
    global compiled_graph, _stdio_ctx, _session_ctx

    logger.info("[close_agent] Shutting down MCP session")
    if _session_ctx:
        await _session_ctx.__aexit__(None, None, None)
        _session_ctx = None

    if _stdio_ctx:
        await _stdio_ctx.__aexit__(None, None, None)
        _stdio_ctx = None

    compiled_graph = None
    logger.info("[close_agent] MCP session closed")
