from fastmcp import FastMCP

from app.core.config import get_conversation_storage_paths
from app.core.security_ast import validate_python, validate_sql
from app.services.duckdb_engine import DuckDBEngine
from app.services.sandbox_runner import SandboxRunner

mcp = FastMCP("DataPal MCP Server")


@mcp.tool()
def inspect_db_schema(conversation_id: str) -> str:
    """
    Returns all table names, column names/types, and 3 sample rows.
    The agent MUST call this before writing any SQL or Python query.
    """
    db_path = get_conversation_storage_paths(conversation_id)["db"]
    if not db_path.exists():
        return "{}"
    with DuckDBEngine(db_path, read_only=True) as db:
        return db.get_schema_summary()


@mcp.tool()
def execute_sql_query(query: str, conversation_id: str) -> dict:
    """
    Executes a read-only SQL query against DuckDB.
    Results are automatically capped at 500 rows.
    Args:
        query: A valid SQL SELECT statement.
        conversation_id: ID of the conversation to query the database for.
    """
    try:
        validate_sql(query)
    except (ValueError, SyntaxError) as e:
        return {"success": False, "error": str(e)}

    db_path = get_conversation_storage_paths(conversation_id)["db"]
    if not db_path.exists():
        return {
            "success": False,
            "error": "Database has not been initialized for this conversation yet.",
        }
    with DuckDBEngine(db_path, read_only=True) as db:
        return db.execute_read_query(query)


@mcp.tool()
def execute_python_analysis(code_str: str, conversation_id: str) -> dict:
    """
    Validates Python code with AST safety checker, then runs it in Docker sandbox.
    Args:
        code_str: Raw Python script string to execute.
        conversation_id: Conversation ID used to scope storage I/O to the correct
            per-conversation folder under storage/<conversation_id>/.
    """
    # AST check — keep this even though the agent's ast_eval_node runs first.
    # The MCP tool is an independent security boundary and should not trust its callers.
    try:
        validate_python(code_str)
    except (ValueError, SyntaxError) as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "exit_code": 1,
            "artifacts": [],
        }

    # Resolve per-conversation storage directories.
    paths = get_conversation_storage_paths(conversation_id)

    # Docker sandbox execution — output artifacts land in
    # storage/<conversation_id>/output/artifacts/
    # Any failure (e.g. Docker daemon unavailable) is returned as a structured
    # result so the agent can react instead of the MCP call erroring out.
    try:
        return SandboxRunner().execute(
            code_str,
            uploads_path=paths["uploads"],
            sql_results_path=paths["sql_results"],
            artifacts_path=paths["artifacts"],
        )
    except Exception as e:  # noqa: BLE001
        return {
            "stdout": "",
            "stderr": f"Sandbox unavailable: {e}",
            "exit_code": 1,
            "artifacts": [],
        }


if __name__ == "__main__":
    mcp.run()
