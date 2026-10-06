import contextlib
import json
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    CodeExecNode,
    CodeGenNode,
    DirectAnswerNode,
    ErrorCorrectionNode,
    FinalFormattingNode,
    PlannerNode,
    SummarizeExecResult,
    advance_step,
    ast_eval_node,
    ast_eval_router,
    code_exec_router,
    code_type_router,
    fallback_failure_node,
    plan_intent,
    step_router,
)
from app.agent.state import AgentState
from app.core import agent_checkpoint
from app.core.logging_config import get_logger
from app.schemas.plan import Plan

load_dotenv()

logger = get_logger(__name__)


async def run_graph(question: str, conversation_id: str) -> dict:
    """
    Invoke the pre-compiled LangGraph state machine and return the final state.

    The MCP subprocess and compiled graph are long-lived (initialised at app
    startup by ``init_agent``); this function simply submits work to them.
    """
    from app.core.agent_runtime import compiled_graph  # noqa: PLC0415
    from app.core.config import get_conversation_storage_paths  # noqa: PLC0415

    if compiled_graph is None:
        raise RuntimeError("Agent runtime is not initialised. Call init_agent() at startup.")

    paths = get_conversation_storage_paths(conversation_id)
    sql_results_dir = paths["sql_results"]
    sql_results_dir.mkdir(parents=True, exist_ok=True)

    # Clean out any leftover intermediate files from a previous run
    for item in sql_results_dir.glob("*"):
        if item.is_file():
            with contextlib.suppress(OSError):
                item.unlink()

    logger.debug("[run_graph] Using intermediate sql_results dir: %s", sql_results_dir)
    initial_state: dict = {
        "conversation_id": conversation_id,
        "messages": [HumanMessage(content=question)],
        "current_question": question,
        "plan": Plan(intent="Code execution", steps=[]),
        "current_step_index": 0,
        "generated_code": "",
        "retry_count": 0,
        "ast_violation": False,
        "final_answer": "",
        "datasets": {},
        "sql_execution_output": None,
        "python_execution_output": None,
        "execution_error": None,
        "sql_results_dir": str(sql_results_dir),
        # schema_context is intentionally omitted: LangGraph will keep the
        # checkpointed value from the previous turn so PlannerNode can reuse
        # it without a redundant MCP call.  On the very first turn the
        # checkpoint has no value and state.get("schema_context") returns
        # None / "", triggering a fresh fetch.
    }

    config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}
    logger.debug("[run_graph] Invoking graph | question=%r", question)
    try:
        result = await compiled_graph.ainvoke(initial_state, config=config)
    finally:
        # Wipe intermediate files so they only live during a single user question
        for item in sql_results_dir.glob("*"):
            if item.is_file():
                with contextlib.suppress(OSError):
                    item.unlink()
        # Reset intermediate datasets in checkpointer state for next turn
        try:
            await compiled_graph.aupdate_state(
                config,
                {
                    "datasets": {},
                    "sql_execution_output": None,
                    "python_execution_output": None,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[run_graph] Failed to reset checkpointed state: %s", exc)

    logger.debug(
        "[run_graph] Graph execution complete — intermediate dir cleaned up | final_answer_len=%d",
        len(result.get("final_answer", "")),
    )
    return result


# ---------------------------------------------------------------------------
# Node-label → human-readable status for SSE progress events
# ---------------------------------------------------------------------------

_NODE_LABELS: dict[str, str] = {
    "planner": "Planning analysis steps…",
    "direct_answer": "Generating direct answer…",
    "code_generation": "Generating code…",
    "ast_evaluation": "Checking code safety…",
    "code_execution": "Executing code…",
    "summarize_execution_result": "Summarising results…",
    "error_correction": "Correcting error…",
    "advance_step": "Advancing to next step…",
    "final_formatting": "Formatting final answer…",
    "fallback_failure": "Handling failure…",
}


async def stream_graph(question: str, conversation_id: str) -> AsyncGenerator[str, None]:
    """
    Async generator that streams SSE-formatted lines while the agent runs.

    Yields lines in the format::

        data: {"type": "progress", "node": "planner", "label": "Planning…"}\\n\\n
        data: {"type": "token",    "content": "…chunk…"}\\n\\n
        data: {"type": "done",     "content": "<full answer>"}\\n\\n

    The caller is responsible for wrapping these in a FastAPI ``StreamingResponse``
    with ``media_type="text/event-stream"``.
    """
    from app.core.agent_runtime import compiled_graph  # noqa: PLC0415
    from app.core.config import get_conversation_storage_paths  # noqa: PLC0415

    if compiled_graph is None:
        raise RuntimeError("Agent runtime is not initialised. Call init_agent() at startup.")

    paths = get_conversation_storage_paths(conversation_id)
    sql_results_dir = paths["sql_results"]
    sql_results_dir.mkdir(parents=True, exist_ok=True)

    # Clean out any leftover intermediate files from a previous run
    for item in sql_results_dir.glob("*"):
        if item.is_file():
            with contextlib.suppress(OSError):
                item.unlink()

    logger.debug("[stream_graph] Using intermediate sql_results dir: %s", sql_results_dir)
    initial_state: dict = {
        "conversation_id": conversation_id,
        "messages": [HumanMessage(content=question)],
        "current_question": question,
        "plan": Plan(intent="Code execution", steps=[]),
        "current_step_index": 0,
        "generated_code": "",
        "retry_count": 0,
        "ast_violation": False,
        "final_answer": "",
        "datasets": {},
        "sql_execution_output": None,
        "python_execution_output": None,
        "execution_error": None,
        "sql_results_dir": str(sql_results_dir),
        # schema_context is intentionally omitted: LangGraph will keep the
        # checkpointed value from the previous turn so PlannerNode can reuse
        # it without a redundant MCP call.  On the very first turn the
        # checkpoint has no value and state.get("schema_context") returns
        # None / "", triggering a fresh fetch.
    }

    final_answer = ""
    config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}

    try:
        async for event in compiled_graph.astream_events(
            initial_state, config=config, version="v2"
        ):
            kind = event.get("event", "")
            name = event.get("name", "")

            # ── Node started → emit a progress tick ───────────────────────
            if kind == "on_chain_start" and name in _NODE_LABELS:
                payload = json.dumps(
                    {"type": "progress", "node": name, "label": _NODE_LABELS[name]}
                )
                yield f"data: {payload}\n\n"

            # ── LLM token delta → stream the text to the client ──────────
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                token = ""
                if chunk is not None:
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    payload = json.dumps({"type": "token", "content": token})
                    yield f"data: {payload}\n\n"

            # ── Graph finished → capture the final answer ─────────────────
            elif kind == "on_chain_end" and name == "LangGraph":
                output = event.get("data", {}).get("output", {})
                final_answer = output.get("final_answer", "")
    finally:
        # Wipe intermediate files so they only live during a single user question
        for item in sql_results_dir.glob("*"):
            if item.is_file():
                with contextlib.suppress(OSError):
                    item.unlink()
        # Reset intermediate datasets in checkpointer state for next turn
        try:
            await compiled_graph.aupdate_state(
                config,
                {
                    "datasets": {},
                    "sql_execution_output": None,
                    "python_execution_output": None,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[stream_graph] Failed to reset checkpointed state: %s", exc)

    # ── Sentinel: done ────────────────────────────────────────────────
    payload = json.dumps({"type": "done", "content": final_answer})
    yield f"data: {payload}\n\n"
    logger.debug(
        "[stream_graph] Stream complete — intermediate dir cleaned up | answer_len=%d",
        len(final_answer),
    )


def build_state_graph(llm, mcp_tools):
    logger.debug("[build_state_graph] Building state graph")
    graph = StateGraph(AgentState)  # type: ignore

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("planner", PlannerNode(llm, mcp_tools))
    graph.add_node("code_generation", CodeGenNode(llm))
    graph.add_node("ast_evaluation", ast_eval_node)
    graph.add_node("code_execution", CodeExecNode(mcp_tools))
    graph.add_node("summarize_execution_result", SummarizeExecResult())
    graph.add_node("advance_step", advance_step)
    graph.add_node("error_correction", ErrorCorrectionNode(llm))
    graph.add_node("direct_answer", DirectAnswerNode(llm))
    graph.add_node("final_formatting", FinalFormattingNode(llm))
    graph.add_node("fallback_failure", fallback_failure_node)
    graph.add_node("code_type_router", lambda state: state)  # passthrough routing node

    graph.set_entry_point("planner")

    # ── Edges ──────────────────────────────────────────────────────────────────
    # planner → intent check
    graph.add_conditional_edges(
        "planner",
        plan_intent,
        {"direct": "direct_answer", "execute": "code_generation"},
    )

    # direct answer path terminates immediately
    graph.add_edge("direct_answer", END)

    # code_gen → safety gate
    graph.add_edge("code_generation", "ast_evaluation")

    # ast_eval → safe: proceed to execution
    #           → retry: violation found, retries remaining — send to error_correction
    #           → give_up: violation found, budget exhausted — terminate
    graph.add_conditional_edges(
        "ast_evaluation",
        ast_eval_router,
        {
            "safe": "code_execution",
            "retry": "error_correction",
            "give_up": "fallback_failure",
        },
    )

    # code_exec → retry / give up / advance to next step
    graph.add_conditional_edges(
        "code_execution",
        code_exec_router,
        {
            "success": "advance_step",
            "retry": "error_correction",
            "give_up": "fallback_failure",
        },
    )

    # error_correction generates corrected code → back to safety gate (skip code_gen re-run)
    graph.add_edge("error_correction", "ast_evaluation")

    # advance_step → loop back for next step or finish
    graph.add_conditional_edges(
        "advance_step",
        step_router,
        {"next_step": "code_generation", "done": "code_type_router"},
    )

    # code_type_router → sql: summarise result then format; python: format directly
    graph.add_conditional_edges(
        "code_type_router",
        code_type_router,
        {
            "sql": "summarize_execution_result",
            "python": "final_formatting",
        },
    )

    # summarize_execution_result → final answer
    graph.add_edge("summarize_execution_result", "final_formatting")

    graph.add_edge("fallback_failure", END)
    graph.add_edge("final_formatting", END)

    compiled = graph.compile(checkpointer=agent_checkpoint.checkpointer)
    logger.debug("[_build_state_graph] Graph compiled with %d nodes", 11)
    return compiled
