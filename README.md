# DataPal

> **Your autonomous AI data analyst for natural language exploration, instant visualizations, and actionable insights.**

DataPal is an intelligent AI data assistant that transforms raw tabular datasets into instant answers, summaries, and charts through natural conversation. Instead of writing complex SQL queries, building custom spreadsheet formulas, or wrestling with manual scripts, users simply upload their files and ask questions in plain English. The platform autonomously inspects the data, devises an analytical plan, writes and executes safe code, and delivers clear insights with visual plots. Whether validating a business hypothesis or discovering hidden trends, DataPal makes data exploration effortless and accessible to everyone.

---

## 💡 Why Use DataPal?

- **Save Hours of Manual Exploration**: Eliminate repetitive data wrangling, formula troubleshooting, and manual spreadsheet slicing. Let the AI agent do the heavy lifting in seconds.
- **Democratize Data Analysis for Non-Technical Users**: Anyone—product managers, marketers, executives, or domain experts—can ask questions and uncover insights without needing to know SQL, Python, or data science libraries.
- **Instant Visual Storytelling**: Go beyond dense tables of numbers. DataPal automatically detects when a visual aids clarity and produces publication-ready charts (bar charts, line graphs, distributions, scatter plots) alongside textual explanations.
- **Self-Healing & Reliable Accuracy**: Unlike generic chatbots that hallucinate metrics or break on syntax errors, DataPal runs actual code directly against your data, validates calculations, and autonomously recovers from errors before presenting results.
- **Context-Aware, Multi-Turn Conversations**: Ask natural follow-up questions such as *"Why did revenue drop in Q3?"* or *"Break that down by category instead."* DataPal retains memory of previous conversation turns and intermediate query results.
- **Enterprise-Grade Security & Sandboxing**: Code generation runs with dual-layer safety—AST guardrails block hazardous operations upfront, and Python execution runs inside an isolated, network-restricted Docker container with strict memory limits.

---

## ✨ Key Features

- 📂 **Tabular Data Ingestion & Management**  
  Easily upload tabular datasets in **CSV** or **Parquet** formats. Uploaded datasets are automatically organized into per-conversation workspaces and ingested into an embedded, ultra-fast **DuckDB** database engine.
- 💬 **Prompt-Driven Autonomous Analysis**  
  Ask analytical questions in natural language. The agent inspects schemas, column types, and sample records, formulates structured analytical plans (identifying target entities, grains, measures, and aggregations), and delivers clean, synthesized answers.
- 💻 **Dynamic Code Generation & Execution**  
  The agent intelligently chooses the right tool for the job: generating high-performance **DuckDB SQL** for relational querying, aggregations, and joins, or generating **Python (Pandas, NumPy, SciPy)** for complex statistics and data manipulations via the Model Context Protocol (MCP).
- 📊 **Automated Visual Chart Creation**  
  When visual presentation helps convey findings, the agent writes Matplotlib / Seaborn plotting scripts to produce visual charts and graphs, exporting them as downloadable and viewable file artifacts.
- 🛠️ **Autonomous Error Correction (Self-Healing Loop)**  
  If generated SQL or Python code encounters a syntax error, schema mismatch, or runtime exception, the agent inspects the stack trace and execution error, adjusts the code, and retries execution automatically (up to retry limits) without requiring manual intervention.
- 🛡️ **AST Guardrails & Sandboxed Docker Execution**  
  Security is built into the core:
  - **AST Validation**: Code is pre-scanned using Python Abstract Syntax Trees and SQL Glot parsers to prohibit dangerous modules (e.g., `os`, `subprocess`, `socket`) and destructive SQL operations (`DROP`, `DELETE`, `UPDATE`).
  - **Docker Sandbox Runner**: Python scripts execute inside a restricted, non-root, read-only Docker container with strict memory caps (512MB), CPU limits, and no network access.
- ⚡ **Real-Time Step Tracking & SSE Streaming**  
  Track the agent's progress in real-time via Server-Sent Events (SSE). The user receives live status updates (*"Planning analysis steps"*, *"Executing code"*, *"Checking code safety"*, *"Summarising results"*) and token-by-token streamed final answers.
- 🔒 **User Authentication & Session Persistence**  
  Includes full JWT-based user authentication and isolated conversation management. LangGraph's PostgreSQL checkpointer persists agent execution states, ensuring reliable multi-turn dialogs across user sessions.

---

## 📁 Project Architecture

```text
datapal/
├── backend/                       # FastAPI backend server & AI agent engine
│   ├── app/
│   │   ├── agent/                 # LangGraph state machine, nodes, and routing
│   │   ├── api/                   # REST API endpoints (Auth, Conversations, Messages, Uploads, Artifacts)
│   │   ├── core/                  # Database connections, AST security, and agent runtime
│   │   ├── mcp/                   # FastMCP server exposing tools (schema, SQL, Python sandbox)
│   │   ├── models/                # SQLAlchemy database models
│   │   ├── prompt/                # Specialized LLM system prompts and guidelines
│   │   ├── schemas/               # Pydantic schemas and validation models
│   │   └── services/              # DuckDB engine and Docker sandbox runner
│   ├── docker/
│   │   └── runner.Dockerfile      # Container definition for the Python execution sandbox
│   ├── storage/                   # Per-conversation workspace storage (uploads, results, artifacts)
│   └── tests/                     # Test suite (pytest)
├── frontend/                      # User Interface application (in active development)
└── README.md                      # Project documentation
```

---

## 🚀 Local Setup & Quickstart Guide

> **Note:** Currently, the backend REST API is available for local setup and testing. The frontend UI is currently under active development. You can interact with and test all features using FastAPI's interactive Swagger UI.

### ⚙️ Prerequisites

Ensure the following tools are installed on your machine:
- **Python**: 3.10+ (Python 3.12 or 3.14 recommended)
- **Docker Desktop**: Installed and running
- **PostgreSQL**: Version 15 or higher (local installation or via Docker)
- **OpenAI API Key**: For agent planning, code generation, and answering

---

### Step 1: Start Docker & Build the Sandbox Runner Image

DataPal runs Python data analysis inside an isolated Docker container. Ensure Docker Desktop is active, then build the sandbox image from the project root:

```bash
docker build -t data-agent-runner:latest -f backend/docker/runner.Dockerfile backend/docker
```

---

### Step 2: Set Up the PostgreSQL Database

DataPal uses PostgreSQL for user management, conversation history, and LangGraph agent checkpoints.

#### Option A: Run PostgreSQL via Docker (Recommended for quick start)
```bash
docker run -d \
  --name datapal-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=123 \
  -e POSTGRES_DB=datapal \
  -p 5432:5432 \
  postgres:16-alpine
```

#### Option B: Use an Existing Local PostgreSQL Server
Ensure your local PostgreSQL service is running and create the `datapal` database:
```sql
CREATE DATABASE datapal;
```

---

### Step 3: Configure Backend Environment Variables

Navigate to the `backend/` directory and configure your `.env` file:

```bash
cd backend
```

Create or edit `backend/.env`:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# PostgreSQL Database Connection URL
DATABASE_URL=postgresql://postgres:123@localhost:5432/datapal

# JWT Authentication
JWT_SECRET_KEY=your_generated_random_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=60

# (Optional) LangSmith Tracing & Observability
LANGSMITH_TRACING=false
# LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# LANGSMITH_API_KEY=your_langsmith_api_key_here
# LANGSMITH_PROJECT="DataPal"
```

---

### Step 4: Set Up Python Virtual Environment & Install Dependencies

From the `backend/` directory:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

### Step 5: Start the Backend Server

Start the FastAPI server using `uvicorn`:

```bash
uvicorn app.main:app --reload
```

Upon startup, DataPal will:
1. Automatically initialize all database tables in PostgreSQL.
2. Setup LangGraph's asynchronous PostgreSQL checkpointer.
3. Start the internal FastMCP server for DuckDB and Docker sandbox operations.

The server will be running at `http://127.0.0.1:8000`.

---

### Step 6: Test & Use the App via Swagger UI

Open your browser to the interactive API documentation:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

#### Quick Walkthrough:
1. **Register & Login** (`/api/v1/auth/register` and `/api/v1/auth/login`): Obtain your JWT bearer token and click **Authorize** at the top of Swagger UI.
2. **Create a Conversation** (`POST /api/v1/conversations/`): Creates an isolated conversation session.
3. **Upload Tabular Data** (`POST /api/v1/conversations/{id}/uploads`): Upload a `.csv` or `.parquet` file. DataPal automatically inspects the schema and ingests it into DuckDB.
4. **Chat & Analyze** (`POST /api/v1/conversations/{id}/messages`): Ask questions like *"What are the top 5 selling items by revenue?"* or *"Plot the sales trend over time"*. Watch the agent plan, query, compute, and stream answers and chart artifacts back!
5. **View Generated Artifacts** (`GET /api/v1/conversations/{id}/artifacts`): Retrieve generated plots, charts, and intermediate query results.

---

## 🧪 Running Tests

To verify that the agent nodes, AST security, and database integrations pass all tests, run:

```bash
cd backend
pytest
```
