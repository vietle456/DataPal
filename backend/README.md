# DataPal Backend

The backend engine for **DataPal**, built with **FastAPI**, **LangGraph**, **DuckDB**, and **Pandas**, featuring AST-based security guardrails and isolated Docker sandboxing for safe code generation and dynamic execution.

---

## 🧠 Agent Architecture

DataPal implements a stateful **Plan-and-Execute** agent architecture orchestrating LLM reasoning, code synthesis, security validation, and deterministic data execution through a **LangGraph** state machine.

![DataPal Agent Architecture](../assets/data_agent_graph.drawio.png)

### Architectural Overview

Traditional ReAct agents often suffer from infinite reasoning loops, context drift, or unpredictable tool chaining when dealing with complex data queries. DataPal addresses this by decoupling high-level analytical planning from concrete code execution through a structured **Plan-and-Execute** workflow:

```text
User Question ──► [Planner] ──┬──► (Direct Answer) ───────────────► Output
                              │
                              └──► (Code Execution Plan)
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
         [Code Generation] ◄───┐                       [Summarize Results]
                 │             │                                 │
                 ▼             │                                 ▼
         [AST Evaluation] ──┐  │                        [Final Formatting]
                 │          │  │                                 │
           (Safe)│  (Retry) │  │                                 ▼
                 ▼          ├──┴── [Error Correction]          Output
         [Code Execution] ──┘
                 │
           (Next Step)
                 ▼
          [Advance Step]
```

### Core Workflow Stages

1. **Analytical Planning (`PlannerNode`)**:
   - **Schema Prefetching**: Automatically inspects table schemas, column data types, and sample records from the conversation's embedded DuckDB database via MCP tools.
   - **Intent Classification**: Classifies queries into:
     - `Direct answer`: Handled immediately without executing code (e.g., conceptual questions, conversational inquiries, or when no schema is present).
     - `Code execution`: Decomposed into an ordered series of discrete execution steps (`SQL_QUERY`, `PYTHON`, `ANSWER`).
   - **Semantic Intent Modeling**: For analytical steps, the planner specifies target entities, grain (entity/row/group), grouping dimensions, measures, aggregation functions, filters, and sorting rules following a *"combine first, split only when necessary"* philosophy.

2. **Context-Aware Code Generation (`CodeGenNode`)**:
   - Generates executable code targeted to the step modality:
     - **DuckDB SQL**: For high-performance relational filtering, joins, groupings, and window aggregations. Results are automatically exported to Parquet.
     - **Python**: For advanced statistical operations, machine learning calculations, or generating visual plots using Pandas, NumPy, SciPy, Matplotlib, and Seaborn.
   - Retains context from prior steps (e.g., referencing intermediate Parquet files produced by preceding SQL steps).

3. **AST Safety Gate (`ast_eval_node`)**:
   - Before execution, code undergoes static analysis using Python's `ast` module and SQLGlot parsers (`SecurityVisitor`).
   - **Blocked Python constructs**: Restricts imports of dangerous modules (`os`, `sys`, `subprocess`, `socket`, `shutil`, `pathlib`), dynamic execution builtins (`eval`, `exec`, `__import__`, `open`), and reflection utilities (`getattr`, `globals`, `locals`).
   - **Blocked SQL statements**: Prohibits mutating operations (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `CREATE`).
   - Safe code proceeds to execution; violations trigger the self-healing loop.

4. **Multi-Engine Execution via FastMCP (`CodeExecNode`)**:
   - Dispatches execution tasks to independent MCP tools:
     - `execute_sql_query`: Runs read-only queries against the conversation's DuckDB database, capturing table previews and persisting intermediate query Parquet files.
     - `execute_python_analysis`: Spawns an isolated **Docker container sandbox** (`data-agent-runner:latest`) with read-only root filesystem, memory limits (512MB), no network access (`network_mode="none"`), and drop-all capabilities (`cap_drop=["ALL"]`).
   - Artifacts (such as PNG charts and plots) generated inside the container are automatically persisted into the conversation's output storage.

5. **Self-Healing & Autonomous Error Correction (`ErrorCorrectionNode`)**:
   - If SQL execution fails, a Python script throws a runtime exception, or the AST validator rejects unsafe syntax, the agent enters a self-healing loop (up to 3 retries).
   - The LLM receives the schema, the failed code, the plan step, and the exact compiler/runtime error message to diagnose and rectify the issue, routing back to AST validation.
   - If the retry budget is exhausted, the workflow transitions safely to `fallback_failure`.

6. **Result Summarization & Response Synthesis (`SummarizeExecResult` & `FinalFormattingNode`)**:
   - For SQL-driven flows, `SummarizeExecResult` formats key aggregations and previews.
   - `FinalFormattingNode` synthesizes numbers, findings, caveats, and generated visual artifacts into a polished, professional natural-language answer.

7. **State Persistence & Real-Time Streaming**:
   - **PostgreSQL Checkpointing**: Agent state is checkpointed per conversation turn using LangGraph's `AsyncPostgresSaver`, enabling seamless multi-turn dialogue and context continuity.
   - **Server-Sent Events (SSE)**: Streams step progress notifications (e.g., *"Planning analysis steps"*, *"Executing code"*, *"Checking code safety"*) and token deltas directly to the client.

---

## ⚙️ Prerequisites

- **Python**: 3.10 or higher (Python 3.12 or 3.14 recommended)
- **Docker Desktop**: Installed and running (required for sandboxed Python code execution)
- **PostgreSQL**: Version 15+ (for metadata, conversation logs, and LangGraph checkpoints)
- **OpenAI API Key**: Required for planner and code generation models

---

## 🚀 Setup & Installation

### 1. Build the Docker Sandbox Image

The Python analysis node executes generated scripts inside an isolated container. From the `backend/` directory, build the image:

```bash
docker build -t data-agent-runner:latest -f docker/runner.Dockerfile docker
```

### 2. Set Up PostgreSQL

Ensure PostgreSQL is running and has a database named `datapal`:

**Using Docker:**
```bash
docker run -d \
  --name datapal-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=123 \
  -e POSTGRES_DB=datapal \
  -p 5432:5432 \
  postgres:16-alpine
```

### 3. Create Virtual Environment

From the `backend/` directory:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

Install the backend package in editable mode:

```bash
pip install -e .
```

*(Or using pip dependencies)*:
```bash
pip install -r requirements.txt
```

### 5. Environment Configuration

Create a `.env` file in `backend/`:

```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# PostgreSQL Database
DATABASE_URL=postgresql://postgres:123@localhost:5432/datapal

# JWT Authentication
JWT_SECRET_KEY=your_random_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Optional LangSmith Tracing
LANGSMITH_TRACING=false
# LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# LANGSMITH_API_KEY=your_langsmith_api_key_here
# LANGSMITH_PROJECT="DataPal"
```

---

## 🧪 Running Tests

Run the backend test suite using `pytest`:

```bash
pytest
```

---

## 🖥️ Running the Server

Start the FastAPI development server:

```bash
uvicorn app.main:app --reload
```

Upon startup, the server automatically initializes database tables and LangGraph checkpointer schemas.

Interactive API documentation will be available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
