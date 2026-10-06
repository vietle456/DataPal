from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.schemas.execution_output import PythonExecutionResult, SQLExecutionResult
from app.schemas.intermediate_artifact import DatasetArtifact
from app.schemas.plan import Plan


class AgentState(TypedDict):
    conversation_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    current_question: str  # the user's question for the current turn only
    plan: Plan
    current_step_index: int
    schema_context: str | None
    schema_version: float | None
    generated_code: str
    retry_count: int
    ast_violation: bool
    datasets: dict[str, DatasetArtifact]
    sql_execution_output: SQLExecutionResult | None
    python_execution_output: PythonExecutionResult | None
    execution_error: str | None
    final_answer: str
    sql_results_dir: str
