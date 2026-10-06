import sys
from pathlib import Path

from mcp import StdioServerParameters

# Root of the backend package (the directory that contains the 'app' package)
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

# MCP server parameters
# sys.executable ensures the subprocess uses the same venv Python as the parent,
# so all dependencies (fastmcp, duckdb, etc.) are available.
# '-m app.mcp.mcp_server' (module mode) also ensures 'app.*' imports resolve
# correctly against the backend directory on sys.path.
MCP_SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "app.mcp.mcp_server"],
    cwd=str(_BACKEND_DIR),
)

# Local storage — base directory for all conversation-scoped sub-folders
STORAGE_ROOT = _BACKEND_DIR / "storage"

# Shared DuckDB database (not conversation-scoped)
DB_PATH = STORAGE_ROOT / "db.duckdb"


def get_conversation_storage_paths(conversation_id: str) -> dict[str, Path]:
    """Return the three storage paths scoped to a specific conversation.

    Directory layout::

        storage/
        └── <conversation_id>/
            ├── input/
            │   └── uploads/          ← uploaded source files
            ├── intermediate/
            │   └── sql_results/      ← parquet exports of SQL query results
            └── output/
                └── artifacts/        ← Python-produced charts, reports, etc.

    The directories are *not* created here; callers are responsible for
    calling ``Path.mkdir(parents=True, exist_ok=True)`` before first use.
    """
    conv_root = STORAGE_ROOT / conversation_id
    return {
        "db": conv_root / "db.duckdb",
        "uploads": conv_root / "input" / "uploads",
        "sql_results": conv_root / "intermediate" / "sql_results",
        "artifacts": conv_root / "output" / "artifacts",
    }
