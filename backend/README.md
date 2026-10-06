# DataPal Backend

The backend for **DataPal**, built with **FastAPI**, **LangGraph**, **DuckDB**, and **Pandas**, featuring AST-based security guardrails for safe dynamic python execution.

---

## ⚙️ Prerequisites

- **Python**: 3.10 or higher
- **Pip**: Latest version

---

## 🚀 Setup & Installation

### 1. Create Virtual Environment

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

### 2. Install Dependencies

Install the backend package in editable mode:

```bash
pip install -e .
```

*(Or using `requirements.txt`)*:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in `backend/` (or copy `.env.example` if available):

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 🧪 Running Tests

Run the test suite using `pytest`:

```bash
pytest
```

---

## 🖥️ Running the Server

Start the FastAPI development server:

```bash
uvicorn app.main:app --reload
```

Interactive API documentation will be available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
