import asyncio
import json
import uuid
from pathlib import Path

import duckdb
import pandas as pd
import sqlglot
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlglot import expressions as exp

from app.agent.state import AgentState
from app.core.config import get_conversation_storage_paths
from app.core.logging_config import get_logger
from app.core.security_ast import validate_python, validate_sql
from app.prompt.prompt import (
    CODE_GEN_PYTHON_SYSTEM_PROMPT,
    CODE_GEN_SQL_SYSTEM_PROMPT,
    ERROR_CORRECTION_SYSTEM_PROMPT,
    FINAL_ANSWER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)
from app.schemas.execution_output import PythonExecutionResult, SQLExecutionResult
from app.schemas.intermediate_artifact import DatasetArtifact
from app.schemas.plan import Plan, PlanStep

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MAX_RETRIES = 3  # maximum code-correction attempts per step
MCP_TOOL_TIMEOUT = 60  # seconds before an MCP tool call is considered hung
RESULT_CHAR_BUDGET = 12000  # max chars (~3-4k tokens) of SQL result rows sent to the final answer


class _Route:
    """Route name constants used by all router functions."""

    DIRECT = "direct"
    EXECUTE = "execute"
    SAFE = "safe"
    RETRY = "retry"
    GIVE_UP = "give_up"
    SUCCESS = "success"
    DONE = "done"
    NEXT_STEP = "next_step"
    SQL = "sql"
    PYTHON = "python"


# ── Nodes ─────────────────────────────────────────────────────────────────────


class PlannerNode:
    """Analyzes question + schema to produce a structured JSON analytical plan."""

    def __init__(self, llm, mcp_tools: list | None = None) -> None:
        self.llm = llm
        self._schema_tool = next(
            (t for t in (mcp_tools or []) if getattr(t, "name", None) == "inspect_db_schema"),
            None,
        )

    async def __call__(self, state: AgentState) -> dict:
        question = state["messages"][-1].content
        logger.debug("[PlannerNode] START | question=%r", question)

        # ── Schema: fingerprint caching via DuckDB file mtime (Strategy 1)
        # Reuse cached value on follow-up turns if database file hasn't changed.
        # Fetch fresh schema if an upload happened, database was updated, or on turn 1.
        conversation_id = state["conversation_id"]
        db_path = get_conversation_storage_paths(conversation_id)["db"]
        current_version = db_path.stat().st_mtime if db_path.exists() else 0.0

        cached_schema: str = state.get("schema_context") or ""
        cached_version = state.get("schema_version")

        if cached_schema and cached_version == current_version:
            logger.debug(
                "[PlannerNode] Schema cache HIT (version=%s) — reusing cached schema (%d chars)",
                current_version,
                len(cached_schema),
            )
            schema_context = cached_schema
        elif self._schema_tool:
            logger.debug(
                "[PlannerNode] Schema cache MISS (old=%s, current=%s) — fetching fresh schema via MCP",
                cached_version,
                current_version,
            )
            schema_context = await self._schema_tool.ainvoke({"conversation_id": conversation_id})
            logger.debug("[PlannerNode] Schema fetched (%d chars)", len(schema_context))
        else:
            logger.warning("[PlannerNode] No schema tool found — proceeding without schema")
            schema_context = ""

        # ainvoke may return a list of content items (LangChain MCP adapter behaviour)
        # — coerce to a plain string before any further processing.
        if not isinstance(schema_context, str):
            schema_context = "\n".join(
                item.text if hasattr(item, "text") else str(item) for item in schema_context
            )
            logger.debug(
                "[PlannerNode] schema_context coerced from list to str (%d chars)",
                len(schema_context),
            )

        # Detect empty schema — an empty JSON object '{}' or whitespace means no tables exist
        _schema_stripped = schema_context.strip()
        schema_is_empty = not _schema_stripped or _schema_stripped in ("{}", "[]")
        if schema_is_empty and schema_context:
            logger.warning(
                "[PlannerNode] Schema returned but contains no tables (%r) — "
                "injecting NO_SCHEMA_AVAILABLE notice into prompt",
                _schema_stripped,
            )

        if schema_is_empty:
            schema_section = (
                "\n\nDatabase schema: NO_SCHEMA_AVAILABLE\n"
                "The database returned no tables or an empty schema. "
                "Any question that requires querying data CANNOT be answered."
            )
        else:
            schema_section = f"\n\nDatabase schema:\n{schema_context}"

        question_with_schema = f"User question: {state['messages'][-1].content}{schema_section}"

        # ── Conversation history: inject the last N prior messages so the planner
        # can handle follow-up questions (e.g. "now filter that by date").  We
        # exclude the final message because it is already embedded in
        # question_with_schema above.  Limit to MAX_HISTORY_MESSAGES to keep the
        # context window bounded.
        MAX_HISTORY_MESSAGES = 10
        prior_messages = state["messages"][:-1]  # all turns except the current one
        if len(prior_messages) > MAX_HISTORY_MESSAGES:
            prior_messages = prior_messages[-MAX_HISTORY_MESSAGES:]
            logger.debug(
                "[PlannerNode] Conversation history trimmed to last %d messages",
                MAX_HISTORY_MESSAGES,
            )
        else:
            logger.debug(
                "[PlannerNode] Injecting %d prior message(s) into planner context",
                len(prior_messages),
            )

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            *prior_messages,
            HumanMessage(content=question_with_schema),
        ]

        logger.debug("[PlannerNode] Calling LLM to generate execution plan")
        plan_response = await self.llm.ainvoke(messages)
        content = plan_response.content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            plan_data = json.loads(content)
            plan = Plan.model_validate(plan_data["plan"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error(
                "[PlannerNode] Failed to parse plan response: %s\nRaw content: %r",
                exc,
                content,
            )
            raise RuntimeError(f"Planner LLM returned an invalid JSON plan: {exc}") from exc

        logger.debug(
            "[PlannerNode] Plan generated | intent=%r, steps=%d",
            plan.intent,
            len(plan.steps),
        )
        for i, step in enumerate(plan.steps):
            logger.debug(
                "[PlannerNode]   Step %d: [%s] %s",
                i + 1,
                step.type,
                step.description,
            )

        return {
            "schema_context": schema_context,
            "schema_version": current_version,
            "messages": [plan_response],
            "plan": plan,
            "current_step_index": 0,
            "retry_count": 0,
            "generated_code": "",
            "datasets": {},
            "sql_execution_output": None,
            "python_execution_output": None,
            "execution_error": None,
        }


class CodeGenNode:
    """Generates SQL or Python code for the current plan step only."""

    def __init__(self, llm) -> None:
        self.llm = llm

    async def __call__(self, state: AgentState) -> dict:
        steps = state["plan"].steps
        index = state["current_step_index"]
        current_step = steps[index]

        logger.debug(
            "[CodeGenNode] START | step=%d/%d type=%r desc=%r",
            index + 1,
            len(steps),
            current_step.type,
            current_step.description,
        )

        # Surface previous execution output so the LLM can reference prior results.
        # Strategy differs by the *previous* step's type:
        #   • SQL_QUERY  → pass a compact dataset reference (filename, schema, path)
        #                  so the LLM can emit `pd.read_parquet(path)` rather than
        #                  receiving hundreds of rows verbatim.
        #   • PYTHON / first step → pass stdout as-is (short text, charts, etc.)
        prev_step = steps[index - 1] if index > 0 else None
        prev_step_type = prev_step.type if prev_step else None

        if prev_step_type == "SQL_QUERY":
            sql_output = state.get("sql_execution_output")
            datasets: dict = state.get("datasets") or {}
            # Use the most-recently added dataset artifact
            artifact = next(iter(datasets.values()), None) if datasets else None
            parquet_filename = next(iter(datasets.keys()), None) if datasets else None

            if artifact and parquet_filename and sql_output:
                col_names = [c.get("name", c) for c in (sql_output.columns or [])]
                # SQL runs on the host (via DuckDB/MCP) → use the local storage_path.
                # Python runs inside Docker → use the container_path (bind-mounted).
                parquet_ref = (
                    artifact.storage_path
                    if current_step.type == "SQL_QUERY"
                    else artifact.container_path
                )
                prev_output_section = f"""\
Previous step result (SQL query — data exported to parquet):
  File name   : {parquet_filename}
  Storage path: {parquet_ref}
  Row count   : {sql_output.row_count}
  Columns     : {col_names}
  Preview     : {sql_output.rows[:10]}

Use `pd.read_parquet("{parquet_ref}")` to load this dataset.\
"""
            else:
                # Artifact or sql_execution_output not found in state (edge case)
                prev_output_section = "Previous step result (SQL query): No output available"
        elif prev_step_type == "PYTHON":
            python_output = state.get("python_execution_output")
            if python_output and python_output.analysis_result is not None:
                analysis_json = json.dumps(python_output.analysis_result, indent=2)
                prev_output_section = (
                    f"Previous step result (Python analysis_result):\n{analysis_json}"
                )
            elif python_output and python_output.stdout:
                prev_output_section = (
                    f"Previous step result (Python script stdout):\n{python_output.stdout}"
                )
            else:
                prev_output_section = "No previous step output"
        else:
            prev_output_section = "No previous step output"

        steps_json = json.dumps([s.model_dump() for s in steps], indent=2)

        semantic_section = ""
        if current_step.semantic_intent:
            si = current_step.semantic_intent
            parts = []
            if si.target_entity:
                parts.append(f"Target entity: {si.target_entity}")
            if si.entity_key:
                parts.append(f"Entity key / identifier: {si.entity_key}")
            if si.grain:
                parts.append(f"Analytical grain: {si.grain}")
            if si.grouping_columns:
                parts.append(f"Grouping columns: {si.grouping_columns}")
            if si.measures:
                parts.append(f"Measures: {si.measures}")
            if si.aggregation_function:
                parts.append(f"Aggregation function: {si.aggregation_function}")
            if si.ranking_order:
                parts.append(f"Ranking order: {si.ranking_order}")
            if si.limit is not None:
                parts.append(f"Limit: {si.limit}")
            if si.filters:
                parts.append(f"Filters: {si.filters}")
            if parts:
                semantic_section = "\n  Semantic intent:\n    " + "\n    ".join(parts)

        context_message = HumanMessage(
            content=f"""
User question:
{state["current_question"]}

Schema:
{state["schema_context"]}

Full plan ({len(steps)} steps):
{steps_json}

Current step to implement (step {index + 1} of {len(steps)}):
  Type: {current_step.type}
  Description: {current_step.description}{semantic_section}

{prev_output_section}
"""
        )

        code_gen_prompt = (
            CODE_GEN_SQL_SYSTEM_PROMPT
            if current_step.type == "SQL_QUERY"
            else CODE_GEN_PYTHON_SYSTEM_PROMPT
        )
        messages = [
            SystemMessage(content=code_gen_prompt),
            context_message,
        ]

        logger.debug("[CodeGenNode] Calling LLM to generate code")
        response = await self.llm.ainvoke(messages)
        generated = response.content.strip()

        # Strip markdown fences that the LLM may emit despite prompt instructions
        if generated.startswith("```"):
            lines = generated.splitlines()
            # Drop the opening fence line (e.g. ```sql / ```python / ```)
            lines = lines[1:]
            # Drop the closing fence line if present
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            generated = "\n".join(lines).strip()

        logger.debug(
            "[CodeGenNode] Code generated (%d chars):\n%s",
            len(generated),
            generated,
        )

        return {"generated_code": generated, "messages": [response]}


class CodeExecNode:
    """Executes a plan step via MCP tools — SQL queries via execute_sql_query,
    Python scripts via execute_python_analysis (Docker sandbox)."""

    def __init__(self, mcp_tools) -> None:
        self.mcp_tools = mcp_tools

    def _get_tool(self, name: str):
        tool = next(
            (t for t in self.mcp_tools if getattr(t, "name", None) == name),
            None,
        )
        if tool is None:
            raise RuntimeError(f"MCP tool '{name}' not found")
        return tool

    async def __call__(self, state: AgentState) -> dict:
        step_index = state["current_step_index"]
        step_num = step_index + 1
        current_step = state["plan"].steps[step_index]
        code = state["generated_code"]
        step_type = current_step.type  # "SQL_QUERY" | "PYTHON"

        logger.debug("[CodeExecNode] START | step=%d | type=%s", step_num, step_type)
        logger.debug("[CodeExecNode] Submitting to executor:\n%s", code)

        dataset_artifact: DatasetArtifact | None = None  # populated by SQL branch only

        if step_type == "SQL_QUERY":
            sql_tool = self._get_tool("execute_sql_query")
            logger.debug("[CodeExecNode] Invoking MCP tool 'execute_sql_query'")
            raw_result = await asyncio.wait_for(
                sql_tool.ainvoke(
                    {
                        "query": code,
                        "conversation_id": state["conversation_id"],
                    }
                ),
                timeout=MCP_TOOL_TIMEOUT,
            )

            # Coerce to str — MCP adapter should return a string, but guard
            # against future adapter changes that may return a list or dict.
            if not isinstance(raw_result, str):
                raw_result = json.dumps(raw_result)
            raw_list = json.loads(raw_result)
            raw_item = raw_list[0]
            sql_result = json.loads(raw_item["text"])

            success = sql_result.get("success", True)
            error_msg = sql_result.get("error", "") if not success else ""
            rows = sql_result.get("rows", sql_result.get("data", []))
            row_count = sql_result["summary"]["row_count"]

            logger.debug(
                "[CodeExecNode] SQL execution complete | success=%s, rows=%s",
                success,
                len(rows) if success else "N/A",
            )
            if error_msg:
                logger.debug("[CodeExecNode] SQL error:\n%s", error_msg)

            parquet_path: str | None = None
            result_id: str | None = None
            if success and rows:
                try:
                    sql_results_dir = Path(state["sql_results_dir"])
                    result_id = str(uuid.uuid4())
                    parquet_filename = f"result_{result_id}.parquet"
                    parquet_path = str(sql_results_dir / parquet_filename)
                    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
                    logger.debug(
                        "[CodeExecNode] SQL result exported to parquet: %s",
                        parquet_path,
                    )
                    dataset_artifact = DatasetArtifact(
                        id=result_id,
                        storage_path=parquet_path,
                        container_path=f"/workspace/intermediate/{parquet_filename}",
                    )
                except OSError as exc:
                    logger.warning(
                        "[CodeExecNode] Failed to export SQL result to parquet: %s",
                        exc,
                    )

            sql_exec_output = SQLExecutionResult(
                id=result_id or "",
                success=success,
                error=error_msg if error_msg else None,
                columns=sql_result.get("columns", []),
                rows=rows,
                row_count=row_count,
                summary=None,
            )

            # Add execution output to messages so the next code_gen step can reference it
            output_message = HumanMessage(
                content=f"Step {step_num} SQL execution: success={success}, rows={len(rows) if rows else 0}"
            )

            result: dict = {
                "sql_execution_output": sql_exec_output,
                "execution_error": error_msg if not success else None,
                "messages": [output_message],
            }

            # Merge the new DatasetArtifact into the state's datasets dict (SQL steps only)
            if dataset_artifact is not None:
                parquet_filename = Path(dataset_artifact.storage_path).name
                result["datasets"] = {
                    **(state.get("datasets") or {}),
                    dataset_artifact.id: dataset_artifact,
                    parquet_filename: dataset_artifact,
                }

            return result

        else:  # PYTHON
            python_tool = self._get_tool("execute_python_analysis")
            logger.debug("[CodeExecNode] Invoking MCP tool 'execute_python_analysis'")
            # Call the MCP tool — goes through the MCP protocol to the server process
            # which runs the code in Docker
            raw_result = await asyncio.wait_for(
                python_tool.ainvoke(
                    {
                        "code_str": code,
                        "conversation_id": state["conversation_id"],
                    }
                ),
                timeout=MCP_TOOL_TIMEOUT,
            )
            # logger.debug("[CodeExecNode] Python raw result: %s", raw_result)

            # Coerce to str — guard against adapter changes.
            if not isinstance(raw_result, str):
                raw_result = json.dumps(raw_result)
            result = json.loads(json.loads(raw_result)[0]["text"])
            logger.debug("[CodeExecNode] Python json loads result: %s", result)

            exit_code = result.get("exit_code", -1)
            stdout = result.get("stdout", "") or ""
            stderr = result.get("stderr", "") or ""
            artifacts = result.get("artifacts", [])
            analysis_result = result.get("analysis_result", None)

            logger.debug(
                "[CodeExecNode] Python execution complete | exit_code=%d, stdout_len=%d, stderr_len=%d, artifacts=%s, has_analysis_result=%s",
                exit_code,
                len(stdout),
                len(stderr),
                artifacts,
                analysis_result is not None,
            )
            if stdout:
                logger.debug(
                    "[CodeExecNode] stdout:\n%s",
                    stdout[:500] + ("..." if len(stdout) > 500 else ""),
                )
            if stderr:
                logger.debug("[CodeExecNode] stderr:\n%s", stderr)
            if analysis_result is not None:
                logger.debug(
                    "[CodeExecNode] analysis_result: %s",
                    json.dumps(analysis_result)[:500],
                )

            python_exec_output = PythonExecutionResult(
                success=exit_code == 0,
                stdout=stdout,  # str, required — already defaulted to "" above
                stderr=stderr if stderr else None,
                artifacts=artifacts,
                analysis_result=analysis_result,
            )

            # Add execution output to messages so the next code_gen step can reference it
            output_message = HumanMessage(content=f"Step {step_num} execution output:\n{stdout}")

            return {
                "python_execution_output": python_exec_output,
                "execution_error": stderr if stderr and exit_code != 0 else None,
                "messages": [output_message],
            }


class ErrorCorrectionNode:
    """Uses the LLM to fix failing code; stores the corrected code directly in state."""

    def __init__(self, llm) -> None:
        self.llm = llm

    async def __call__(self, state: AgentState) -> dict:
        steps = state["plan"].steps
        current_step = steps[state["current_step_index"]]
        error = state["execution_error"]
        retry_count = state["retry_count"] + 1

        logger.debug(
            "[ErrorCorrectionNode] START | step=%d, attempt=%d/3",
            state["current_step_index"] + 1,
            retry_count,
        )
        logger.debug("[ErrorCorrectionNode] Error to fix:\n%s", error)

        messages = [
            SystemMessage(content=ERROR_CORRECTION_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""
Original question:
{state["current_question"]}

Schema:
```
{state["schema_context"]}
```

Step being corrected:
  Type: {current_step.type}
  Description: {current_step.description}

Code that was executed:
```
{state["generated_code"]}
```

Error (attempt {retry_count} of 3):
```
{error}
```
"""
            ),
        ]

        logger.debug("[ErrorCorrectionNode] Calling LLM for corrected code")
        response = await self.llm.ainvoke(messages)
        corrected = response.content.strip()

        # Strip markdown fences that the LLM may emit despite prompt instructions
        if corrected.startswith("```"):
            lines = corrected.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            corrected = "\n".join(lines).strip()

        logger.debug(
            "[ErrorCorrectionNode] Corrected code (%d chars):\n%s",
            len(corrected),
            corrected,
        )

        return {
            "retry_count": retry_count,
            "generated_code": corrected,
            "messages": [response],
        }


class DirectAnswerNode:
    """Handles 'Direct answer' intent — synthesizes a response without any code execution."""

    def __init__(self, llm) -> None:
        self.llm = llm

    async def __call__(self, state: AgentState) -> dict:
        # Find the ANSWER step — for direct answers the planner should emit one,
        # but fall back to the first step if the plan is unexpectedly malformed.
        answer_step = next(
            (s for s in state["plan"].steps if s.type == "ANSWER"),
            state["plan"].steps[0],
        )

        logger.debug(
            "[DirectAnswerNode] START | guidance=%r",
            answer_step.description[:120],
        )

        messages = [
            SystemMessage(content=FINAL_ANSWER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""
User question:
{state["current_question"]}

Guidance:
{answer_step.description}
"""
            ),
        ]

        logger.debug("[DirectAnswerNode] Calling LLM for direct answer")
        response = await self.llm.ainvoke(messages)

        logger.debug("[DirectAnswerNode] Answer generated (%d chars)", len(response.content))

        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content)],
        }


class FinalFormattingNode:
    """Synthesizes all execution outputs into a professional final answer via LLM."""

    def __init__(self, llm) -> None:
        self.llm = llm

    async def __call__(self, state: AgentState) -> dict:
        # Determine the last executed step type to pick the right execution result.
        steps = state["plan"].steps
        last_exec_step = next(
            (s for s in reversed(steps) if s.type in ("SQL_QUERY", "PYTHON")),
            None,
        )
        last_step_type = last_exec_step.type if last_exec_step else None

        if last_step_type == "PYTHON":
            python_output = state.get("python_execution_output")
            if python_output and python_output.analysis_result is not None:
                execution_context_label = (
                    "Analysis result (structured findings from Python execution):"
                )
                execution_context = json.dumps(python_output.analysis_result, indent=2)
            elif python_output and python_output.artifacts:
                execution_context_label = "Artifacts generated by Python execution:"
                execution_context = "\n".join(python_output.artifacts)
            else:
                execution_context_label = "Python execution output:"
                execution_context = (python_output.stdout if python_output else None) or "No output"
            logger.debug(
                "[FinalFormattingNode] Python path | has_analysis_result=%s, artifacts=%s",
                python_output.analysis_result is not None if python_output else False,
                python_output.artifacts if python_output else [],
            )
        else:  # SQL_QUERY or fallback
            sql_output = state.get("sql_execution_output")
            execution_context_label = "Summary of SQL execution results:"
            execution_context = (sql_output.summary if sql_output else None) or ""
            logger.debug(
                "[FinalFormattingNode] SQL path | summary_len=%d",
                len(execution_context),
            )

        # Retrieve the planner's ANSWER step description as synthesis guidance.
        # This tells the LLM *what* the answer should address, preventing it from
        # claiming data is missing when the result set is intentionally scoped.
        answer_guidance = next((s.description for s in steps if s.type == "ANSWER"), "")
        guidance_section = (
            f"\nPlanner guidance (what the answer must address):\n{answer_guidance}\n"
            if answer_guidance
            else ""
        )

        messages = [
            SystemMessage(content=FINAL_ANSWER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""
User question:
{state["current_question"]}
{guidance_section}
{execution_context_label}
{execution_context}
"""
            ),
        ]

        logger.debug("[FinalFormattingNode] Calling LLM to synthesize final answer")
        response = await self.llm.ainvoke(messages)

        logger.debug(
            "[FinalFormattingNode] Final answer generated (%d chars)",
            len(response.content),
        )

        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content)],
        }


class SummarizeExecResult:
    """Prepares the SQL execution result for the final answer node.

    Rows are serialised as compact CSV (cheaper than JSON, keys not repeated).

    - Fits within RESULT_CHAR_BUDGET : the full result is passed as CSV.
    - Exceeds the budget             : the first and last rows (as many as fit
      in the budget) are passed as CSV, followed by a DuckDB SUMMARIZE of the
      whole result, plus a note stating how many rows were shown. This keeps
      entity/value pairs (e.g. top publisher + count) visible to the LLM.
    """

    def __call__(self, state: AgentState) -> dict:
        sql_execution_output = state.get("sql_execution_output")
        if sql_execution_output is None:
            # This node is only reached after a successful SQL execution, so
            # this path should never be hit. Log and return a no-op update.
            logger.warning("[SummarizeExecResult] Called with no sql_execution_output — skipping")
            return {}

        rows = sql_execution_output.rows or []
        row_count = sql_execution_output.row_count or len(rows)

        logger.debug("[SummarizeExecResult] START | row_count=%d", row_count)

        full_csv = pd.DataFrame(rows).to_csv(index=False)
        dup_note = ""
        if rows and len(rows) > 1:
            for k in rows[0]:
                vals = [r[k] for r in rows if r.get(k) is not None]
                if vals and all(isinstance(v, str) for v in vals) and len(vals) != len(set(vals)):
                    dup_names = [v for v in set(vals) if vals.count(v) > 1]
                    dup_note = (
                        f"\n[Note on Data Granularity: Column '{k}' contains repeated values ({dup_names}). "
                        f"These represent individual sub-records/breakdowns for the same entity rather than distinct entities.]\n"
                    )
                    break

        if len(full_csv) <= RESULT_CHAR_BUDGET:
            logger.debug("[SummarizeExecResult] Result fits budget — passing full rows as CSV")
            result = full_csv + dup_note
        else:
            head_n, tail_n = self._select_head_tail(full_csv, len(rows))
            header = full_csv.splitlines()[0]
            lines = full_csv.splitlines()[1:]
            head_lines = lines[:head_n]
            tail_lines = lines[len(lines) - tail_n :] if tail_n else []
            omitted = len(lines) - head_n - tail_n

            parts = [
                f"Result has {row_count} rows; showing the first {head_n} and last {tail_n} "
                f"({omitted} omitted from the middle). Row order is as returned by the query.",
                "FIRST ROWS (CSV):",
                "\n".join([header, *head_lines]),
            ]
            if tail_lines:
                parts += ["LAST ROWS (CSV):", "\n".join([header, *tail_lines])]
            parts += [
                "COLUMN STATISTICS OVER ALL ROWS (DuckDB SUMMARIZE):",
                self._summarize(state, sql_execution_output, row_count),
            ]
            result = "\n".join(parts) + dup_note

        logger.debug("[SummarizeExecResult] final_execution_result set (%d chars)", len(result))
        sql_execution_output.summary = result
        return {"sql_execution_output": sql_execution_output}

    @staticmethod
    def _select_head_tail(full_csv: str, total_rows: int) -> tuple[int, int]:
        """Pick how many head and tail rows fit in the budget (head gets 2/3)."""
        lines = full_csv.splitlines()[1:]
        head_budget = int(RESULT_CHAR_BUDGET * 2 / 3)
        tail_budget = RESULT_CHAR_BUDGET - head_budget

        head_n, used = 0, 0
        for line in lines:
            if used + len(line) + 1 > head_budget:
                break
            used += len(line) + 1
            head_n += 1

        tail_n, used = 0, 0
        for line in reversed(lines[head_n:]):
            if used + len(line) + 1 > tail_budget:
                break
            used += len(line) + 1
            tail_n += 1
        return max(head_n, 1) if total_rows else 0, tail_n

    @staticmethod
    def _summarize(state: AgentState, sql_execution_output, row_count: int) -> str:
        dataset = state["datasets"].get(sql_execution_output.id)
        if dataset is None:
            logger.warning(
                "[SummarizeExecResult] Dataset %r not found in state", sql_execution_output.id
            )
            return "(unavailable)"
        try:
            summary_rel = duckdb.sql(
                f"SUMMARIZE SELECT * FROM read_parquet('{dataset.storage_path}');"
            )
            return summary_rel.df().to_string(index=False)
        except (OSError, ValueError, AttributeError, duckdb.Error) as exc:
            logger.warning("[SummarizeExecResult] DuckDB SUMMARIZE failed (%s)", exc)
            col_names = [c.get("name", str(c)) for c in (sql_execution_output.columns or [])]
            return f"(unavailable) {row_count} rows. Columns: {col_names}"


def validate_semantic_sql(query: str, step: PlanStep) -> None:
    """Lightweight semantic validation between plan and generated SQL.

    Checks that if the step's semantic intent specifies entity-level aggregation or distinct
    entity counting, the generated SQL contains the necessary GROUP BY and aggregation
    clauses rather than ranking unaggregated physical rows.
    """
    if not step.semantic_intent:
        return

    si = step.semantic_intent
    try:
        tree = sqlglot.parse_one(query, dialect="duckdb")
    except Exception:
        return  # Syntax errors are caught by validate_sql

    if not isinstance(tree, exp.Query):
        return

    has_group = tree.args.get("group") is not None
    has_limit = tree.args.get("limit") is not None

    # Check 1: Entity-level ranking without GROUP BY
    is_entity_ranking = si.grain == "entity" or (
        si.entity_key
        and si.aggregation_function in ("SUM", "AVG", "COUNT", "COUNT_DISTINCT")
        and (si.limit is not None or has_limit)
    )
    if is_entity_ranking and has_limit and not has_group:
        entity_name = si.entity_key or si.target_entity or "entity"
        agg_fn = si.aggregation_function if si.aggregation_function not in (None, "NONE") else "SUM"
        raise ValueError(
            f"Semantic Validation Error: Step requires entity-level ranking for '{entity_name}' with {agg_fn}() aggregation, "
            f"but the generated query lacks a GROUP BY clause and ranks raw rows with LIMIT. "
            f"Please aggregate using GROUP BY {entity_name} and {agg_fn}(<measure>) before applying ORDER BY and LIMIT."
        )

    # Check 2: Distinct entity counting
    if si.aggregation_function == "COUNT_DISTINCT" and si.entity_key:
        count_calls = list(tree.find_all(exp.Count))
        if count_calls:
            has_distinct_count = any(bool(c.find(exp.Distinct)) for c in count_calls)
            if not has_distinct_count:
                raise ValueError(
                    f"Semantic Validation Error: Step requires counting distinct entities ('{si.entity_key}'), "
                    f"but COUNT was used without DISTINCT. Use COUNT(DISTINCT {si.entity_key}) to avoid overcounting."
                )


def ast_eval_node(state: AgentState) -> dict:
    """Validates generated code safety before sandbox execution."""
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    code_type = current_step.type

    logger.debug("[ast_eval_node] Validating generated code with AST security checker")
    try:
        if code_type == "SQL_QUERY":
            validate_sql(state["generated_code"])
            validate_semantic_sql(state["generated_code"], current_step)
        elif code_type == "PYTHON":
            validate_python(state["generated_code"])
        else:
            raise ValueError(f"Unknown code type: {code_type!r}")
        logger.debug("[ast_eval_node] Code passed AST validation — safe to execute")
        return {"ast_violation": False}
    except SyntaxError as e:
        # ast.parse() raises SyntaxError on malformed code (e.g. leftover markdown fences).
        # Treat this as a correctable violation so the graph routes to error_correction.
        logger.warning("[ast_eval_node] SyntaxError during AST parse: %s", e)
        return {
            "ast_violation": True,
            "execution_error": f"SyntaxError in generated code: {e!s}",
        }
    except ValueError as e:
        logger.warning("[ast_eval_node] AST VIOLATION detected: %s", e)
        return {
            "ast_violation": True,
            "execution_error": f"AST Security Violation: {e!s}",
        }


def advance_step(state: AgentState) -> dict:
    """Increments the step index and resets the retry counter after a successful execution."""
    new_index = state["current_step_index"] + 1
    total_steps = len(state["plan"].steps)
    logger.debug(
        "[advance_step] Step %d/%d complete — moving to step index %d",
        state["current_step_index"] + 1,
        total_steps,
        new_index,
    )
    return {
        "current_step_index": new_index,
        "retry_count": 0,
    }


def fallback_failure_node(state: AgentState) -> dict:  # pylint: disable=unused-argument
    """Terminal node when all retries are exhausted."""
    logger.error(
        "[fallback_failure_node] All retries exhausted for step %d — giving up",
        state["current_step_index"] + 1,
    )
    return {
        "final_answer": f"I was unable to complete this analysis after {MAX_RETRIES} attempts. Please rephrase your question.",
        "messages": [AIMessage(content="Analysis failed after maximum retries.")],
    }


# ── Router functions ──────────────────────────────────────────────────────────


def plan_intent(state: AgentState) -> str:
    """After planning: route to direct answer or start the code execution loop."""
    intent = state["plan"].intent
    route = _Route.DIRECT if intent == "Direct answer" else _Route.EXECUTE
    logger.debug("[plan_intent] intent=%r → route=%r", intent, route)
    return route


def ast_eval_router(state: AgentState) -> str:
    """After AST eval: route to code execution, error correction, or give up.

    Returns:
        "safe"     - code passed validation, proceed to code_exec
        "retry"    - violation found but retries remain, send to error_correction
        "give_up"  - violation found and retry budget exhausted, send to fallback
    """
    if not state.get("ast_violation", False):
        logger.debug("[ast_eval_router] ast_violation=False → safe")
        return _Route.SAFE

    retry_count = state["retry_count"]
    route = _Route.RETRY if retry_count < MAX_RETRIES else _Route.GIVE_UP
    logger.debug(
        "[ast_eval_router] ast_violation=True, retry_count=%d → %r",
        retry_count,
        route,
    )
    return route


def code_exec_router(state: AgentState) -> str:
    """After code execution: advance on success, retry on failure, or give up."""
    sql_out = state.get("sql_execution_output")
    python_out = state.get("python_execution_output")
    exec_out = sql_out or python_out
    success = exec_out.success if exec_out is not None else False
    retry_count = state["retry_count"]

    if success:
        route = _Route.SUCCESS
    elif retry_count < MAX_RETRIES:
        route = _Route.RETRY
    else:
        route = _Route.GIVE_UP

    logger.debug(
        "[code_exec_router] retry_count=%d → route=%r",
        retry_count,
        route,
    )
    return route


def step_router(state: AgentState) -> str:
    """After advancing the step index: loop back to code_gen or finish."""
    steps = state["plan"].steps
    index = state["current_step_index"]

    if index >= len(steps):
        logger.debug("[step_router] All steps complete → done")
        return _Route.DONE

    # ANSWER steps are synthesized by final_formatting, not executed as code
    if steps[index].type == "ANSWER":
        logger.debug("[step_router] Next step is ANSWER type → done (final_formatting)")
        return _Route.DONE

    logger.debug(
        "[step_router] Continuing to step %d/%d → next_step",
        index + 1,
        len(steps),
    )
    return _Route.NEXT_STEP


def code_type_router(state: AgentState) -> str:
    """
    Routes based on the type of the last executed step, called after all steps
    have completed (step_router → done).

    - SQL_QUERY → summarize_execution_result → final_formatting
    - PYTHON    → final_formatting directly

    "ANSWER" steps are consumed by step_router before we reach here; encountering
    one is a programming error.
    """
    # current_step_index points at/past the ANSWER step once all steps are done,
    # so walk backwards to find the last actually-executed (SQL_QUERY / PYTHON) step.
    steps = state["plan"].steps
    executable_types = {"SQL_QUERY", "PYTHON"}
    last_exec_step = next(
        (s for s in reversed(steps) if s.type in executable_types),
        None,
    )

    if last_exec_step is None:
        raise ValueError(
            "[code_type_router] No executable step (SQL_QUERY or PYTHON) found in plan"
        )

    code_type = last_exec_step.type
    if code_type == "SQL_QUERY":
        route = _Route.SQL
    elif code_type == "PYTHON":
        route = _Route.PYTHON
    else:
        raise ValueError(
            f"[code_type_router] Unexpected step type {code_type!r} — "
            "only SQL_QUERY and PYTHON are executable"
        )

    logger.debug(
        "[code_type_router] code_type=%r → route=%r",
        code_type,
        route,
    )
    return route
