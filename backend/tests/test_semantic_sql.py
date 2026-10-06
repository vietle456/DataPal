"""Tests for semantic SQL planning, code generation, and validation.

Verifies domain-agnostic entity-level reasoning across:
1. Entity-level ranking on multi-row datasets (aggregating by entity key with SUM + GROUP BY)
2. Row-level ranking (querying individual transactions/events without GROUP BY)
3. Grouped aggregation (breakdown by dimension with SUM + GROUP BY)
4. Entity counting (counting distinct entities vs counting physical rows)
5. Conversational reference resolution ("their", "during that same period")
6. Explicit granularity (ranking by variant rather than collapsing to parent product)
7. Current regression scenario (vgsales Q1 publisher title count & Q2 top 3 games without duplicate platform records)
8. AST semantic validation (flagging unaggregated entity ranking and missing DISTINCT)
"""

import json
import os
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.nodes import CodeGenNode, validate_semantic_sql
from app.prompt.prompt import PLANNER_SYSTEM_PROMPT
from app.schemas.plan import Plan, PlanStep, SemanticIntent
from app.services.duckdb_engine import DuckDBEngine

HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Unit tests for validate_semantic_sql (Layer 2 AST Semantic Safeguard)
# ---------------------------------------------------------------------------


def test_semantic_validator_catches_unaggregated_entity_ranking():
    """Flags unaggregated entity ranking where LIMIT is used without GROUP BY."""
    step = PlanStep(
        type="SQL_QUERY",
        description="Select top 3 products by sales",
        semantic_intent=SemanticIntent(
            target_entity="product",
            entity_key="product_name",
            grain="entity",
            grouping_columns=["product_name"],
            measures=["sales"],
            aggregation_function="SUM",
            limit=3,
        ),
    )
    bad_query = "SELECT product_name, sales FROM sales_data ORDER BY sales DESC LIMIT 3;"
    with pytest.raises(ValueError, match="Semantic Validation Error.*GROUP BY"):
        validate_semantic_sql(bad_query, step)


def test_semantic_validator_passes_aggregated_entity_ranking():
    """Passes valid entity-level ranking with GROUP BY and SUM."""
    step = PlanStep(
        type="SQL_QUERY",
        description="Select top 3 products by sales",
        semantic_intent=SemanticIntent(
            target_entity="product",
            entity_key="product_name",
            grain="entity",
            grouping_columns=["product_name"],
            measures=["sales"],
            aggregation_function="SUM",
            limit=3,
        ),
    )
    good_query = (
        "SELECT product_name, SUM(sales) AS total_sales "
        "FROM sales_data GROUP BY product_name ORDER BY total_sales DESC LIMIT 3;"
    )
    # Should not raise
    validate_semantic_sql(good_query, step)


def test_semantic_validator_allows_row_level_ranking():
    """Allows row-level queries without GROUP BY when grain='row'."""
    step = PlanStep(
        type="SQL_QUERY",
        description="Top 5 highest transaction amounts",
        semantic_intent=SemanticIntent(
            target_entity="transaction",
            entity_key=None,
            grain="row",
            grouping_columns=[],
            measures=["amount"],
            aggregation_function="NONE",
            limit=5,
        ),
    )
    query = "SELECT transaction_id, amount FROM transactions ORDER BY amount DESC LIMIT 5;"
    # Should not raise
    validate_semantic_sql(query, step)


def test_semantic_validator_catches_missing_distinct_for_entity_counting():
    """Flags COUNT without DISTINCT when step requires counting distinct entities."""
    step = PlanStep(
        type="SQL_QUERY",
        description="Count distinct titles per publisher",
        semantic_intent=SemanticIntent(
            target_entity="title",
            entity_key="Name",
            grain="group",
            grouping_columns=["Publisher"],
            measures=["Name"],
            aggregation_function="COUNT_DISTINCT",
            limit=None,
        ),
    )
    bad_query = (
        "SELECT Publisher, COUNT(Name) AS title_count FROM vgsales GROUP BY Publisher;"
    )
    with pytest.raises(ValueError, match="Semantic Validation Error.*DISTINCT"):
        validate_semantic_sql(bad_query, step)


def test_semantic_validator_passes_distinct_entity_counting():
    """Passes COUNT(DISTINCT col) when step requires counting distinct entities."""
    step = PlanStep(
        type="SQL_QUERY",
        description="Count distinct titles per publisher",
        semantic_intent=SemanticIntent(
            target_entity="title",
            entity_key="Name",
            grain="group",
            grouping_columns=["Publisher"],
            measures=["Name"],
            aggregation_function="COUNT_DISTINCT",
            limit=None,
        ),
    )
    good_query = (
        "SELECT Publisher, COUNT(DISTINCT Name) AS title_count "
        "FROM vgsales GROUP BY Publisher;"
    )
    # Should not raise
    validate_semantic_sql(good_query, step)


# ---------------------------------------------------------------------------
# End-to-end / LLM Semantic Tests (Required 7 Scenarios)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key required for LLM tests")
@pytest.mark.asyncio
async def test_1_entity_level_ranking():
    """Test 1 — Entity-level ranking on a dataset where multiple rows represent the same entity.

    Question: "What are the top 3 products by total sales?"
    Dataset: Products sold across multiple stores and dates.
    Expected: GROUP BY product, SUM(sales), ORDER BY DESC, LIMIT 3.
    """
    schema = """
{
  "store_sales": {
    "columns": [
      {"name": "transaction_id", "type": "BIGINT"},
      {"name": "product_name", "type": "VARCHAR"},
      {"name": "store_id", "type": "VARCHAR"},
      {"name": "sale_date", "type": "DATE"},
      {"name": "revenue", "type": "DOUBLE"}
    ],
    "sample_rows": [
      {"transaction_id": 1, "product_name": "Widget A", "store_id": "S1", "sale_date": "2023-01-01", "revenue": 100.0},
      {"transaction_id": 2, "product_name": "Widget A", "store_id": "S2", "sale_date": "2023-01-02", "revenue": 150.0},
      {"transaction_id": 3, "product_name": "Widget B", "store_id": "S1", "sale_date": "2023-01-01", "revenue": 200.0}
    ]
  }
}
"""
    q = "What are the top 3 products by total sales?"
    async with httpx.AsyncClient() as client:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, http_async_client=client)
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User question: {q}\n\nDatabase schema:\n{schema}"),
        ]
        resp = await llm.ainvoke(messages)
        content = resp.content.strip("`").removeprefix("json").strip()
        plan_dict = json.loads(content)["plan"]
        plan = Plan.model_validate(plan_dict)

        sql_step = plan.steps[0]
        assert sql_step.type == "SQL_QUERY"
        assert sql_step.semantic_intent is not None
        assert sql_step.semantic_intent.grain == "entity"
        assert sql_step.semantic_intent.aggregation_function == "SUM"
        assert "product_name" in (sql_step.semantic_intent.grouping_columns or [sql_step.semantic_intent.entity_key])

        # Now generate SQL from this step
        codegen = CodeGenNode(llm)
        state = {
            "current_question": q,
            "schema_context": schema,
            "plan": plan,
            "current_step_index": 0,
            "datasets": {},
        }
        result = await codegen(state)
        query = result["generated_code"].upper()
        assert "GROUP BY" in query
        assert "PRODUCT_NAME" in query
        assert "SUM(" in query
        assert "LIMIT 3" in query


@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key required for LLM tests")
@pytest.mark.asyncio
async def test_2_row_level_ranking():
    """Test 2 — Row-level ranking without unnecessary grouping.

    Question: "What are the 3 transactions with the highest sales amount?"
    Expected: ORDER BY amount DESC, LIMIT 3 (no GROUP BY).
    """
    schema = """
{
  "transactions": {
    "columns": [
      {"name": "transaction_id", "type": "VARCHAR"},
      {"name": "customer_id", "type": "VARCHAR"},
      {"name": "sales_amount", "type": "DOUBLE"},
      {"name": "created_at", "type": "TIMESTAMP"}
    ],
    "sample_rows": [
      {"transaction_id": "T1", "customer_id": "C1", "sales_amount": 500.0, "created_at": "2023-01-01 10:00:00"},
      {"transaction_id": "T2", "customer_id": "C2", "sales_amount": 1200.0, "created_at": "2023-01-01 11:00:00"}
    ]
  }
}
"""
    q = "What are the 3 transactions with the highest sales amount?"
    async with httpx.AsyncClient() as client:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, http_async_client=client)
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User question: {q}\n\nDatabase schema:\n{schema}"),
        ]
        resp = await llm.ainvoke(messages)
        content = resp.content.strip("`").removeprefix("json").strip()
        plan_dict = json.loads(content)["plan"]
        plan = Plan.model_validate(plan_dict)

        sql_step = plan.steps[0]
        assert sql_step.type == "SQL_QUERY"
        assert sql_step.semantic_intent is not None
        assert sql_step.semantic_intent.grain == "row"

        # Generate SQL
        codegen = CodeGenNode(llm)
        state = {
            "current_question": q,
            "schema_context": schema,
            "plan": plan,
            "current_step_index": 0,
            "datasets": {},
        }
        result = await codegen(state)
        query = result["generated_code"].upper()
        assert "GROUP BY" not in query
        assert "SALES_AMOUNT" in query
        assert "DESC" in query
        assert "LIMIT 3" in query


@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key required for LLM tests")
@pytest.mark.asyncio
async def test_3_grouped_aggregation():
    """Test 3 — Grouped aggregation by dimension.

    Question: "Which regions generated the most revenue?"
    Expected: GROUP BY region, SUM(revenue), ORDER BY DESC.
    """
    schema = """
{
  "regional_sales": {
    "columns": [
      {"name": "order_id", "type": "BIGINT"},
      {"name": "region", "type": "VARCHAR"},
      {"name": "revenue", "type": "DOUBLE"}
    ],
    "sample_rows": [
      {"order_id": 1, "region": "North", "revenue": 100.0},
      {"order_id": 2, "region": "South", "revenue": 200.0}
    ]
  }
}
"""
    q = "Which regions generated the most revenue?"
    async with httpx.AsyncClient() as client:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, http_async_client=client)
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User question: {q}\n\nDatabase schema:\n{schema}"),
        ]
        resp = await llm.ainvoke(messages)
        content = resp.content.strip("`").removeprefix("json").strip()
        plan_dict = json.loads(content)["plan"]
        plan = Plan.model_validate(plan_dict)

        sql_step = plan.steps[0]
        assert sql_step.type == "SQL_QUERY"
        assert sql_step.semantic_intent is not None
        assert "region" in sql_step.semantic_intent.grouping_columns
        assert sql_step.semantic_intent.aggregation_function == "SUM"

        # Generate SQL
        codegen = CodeGenNode(llm)
        state = {
            "current_question": q,
            "schema_context": schema,
            "plan": plan,
            "current_step_index": 0,
            "datasets": {},
        }
        result = await codegen(state)
        query = result["generated_code"].upper()
        assert "GROUP BY" in query
        assert "REGION" in query
        assert "SUM(" in query


@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key required for LLM tests")
@pytest.mark.asyncio
async def test_4_entity_counting():
    """Test 4 — Entity counting distinctness.

    Question: "Which suppliers provide the most distinct products?"
    Expected: GROUP BY supplier, COUNT(DISTINCT product_id) or COUNT(DISTINCT product_name).
    """
    schema = """
{
  "inventory": {
    "columns": [
      {"name": "inventory_id", "type": "BIGINT"},
      {"name": "supplier_name", "type": "VARCHAR"},
      {"name": "product_name", "type": "VARCHAR"},
      {"name": "warehouse_location", "type": "VARCHAR"},
      {"name": "quantity", "type": "INTEGER"}
    ],
    "sample_rows": [
      {"inventory_id": 1, "supplier_name": "Acme", "product_name": "Gadget 1", "warehouse_location": "East", "quantity": 10},
      {"inventory_id": 2, "supplier_name": "Acme", "product_name": "Gadget 1", "warehouse_location": "West", "quantity": 20},
      {"inventory_id": 3, "supplier_name": "Beta", "product_name": "Gadget 2", "warehouse_location": "East", "quantity": 15}
    ]
  }
}
"""
    q = "Which suppliers provide the most distinct products?"
    async with httpx.AsyncClient() as client:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, http_async_client=client)
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User question: {q}\n\nDatabase schema:\n{schema}"),
        ]
        resp = await llm.ainvoke(messages)
        content = resp.content.strip("`").removeprefix("json").strip()
        plan_dict = json.loads(content)["plan"]
        plan = Plan.model_validate(plan_dict)

        sql_step = plan.steps[0]
        assert sql_step.type == "SQL_QUERY"
        assert sql_step.semantic_intent is not None
        assert sql_step.semantic_intent.aggregation_function == "COUNT_DISTINCT"

        # Generate SQL
        codegen = CodeGenNode(llm)
        state = {
            "current_question": q,
            "schema_context": schema,
            "plan": plan,
            "current_step_index": 0,
            "datasets": {},
        }
        result = await codegen(state)
        query = result["generated_code"].upper()
        assert "GROUP BY" in query
        assert "SUPPLIER_NAME" in query
        assert "COUNT(DISTINCT" in query


@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key required for LLM tests")
@pytest.mark.asyncio
async def test_5_conversational_reference():
    """Test 5 — Contextual reference resolution ('their', 'during that same period').

    Turn 1: "Which manufacturer produced the most electric vehicle models in 2022?"
    Assistant: "Tesla produced the most electric vehicle models in 2022."
    Turn 2: "What were their top 3 best-selling models during that same period?"
    Expected: Resolves "their" -> Tesla, "that same period" -> 2022.
    """
    schema = """
{
  "ev_sales": {
    "columns": [
      {"name": "id", "type": "BIGINT"},
      {"name": "manufacturer", "type": "VARCHAR"},
      {"name": "model_name", "type": "VARCHAR"},
      {"name": "trim", "type": "VARCHAR"},
      {"name": "year", "type": "BIGINT"},
      {"name": "units_sold", "type": "INTEGER"}
    ],
    "sample_rows": [
      {"id": 1, "manufacturer": "Tesla", "model_name": "Model 3", "trim": "Standard", "year": 2022, "units_sold": 50000},
      {"id": 2, "manufacturer": "Tesla", "model_name": "Model 3", "trim": "Performance", "year": 2022, "units_sold": 20000}
    ]
  }
}
"""
    q1 = "Which manufacturer produced the most electric vehicle models in 2022?"
    q2 = "What were their top 3 best-selling models during that same period?"

    async with httpx.AsyncClient() as client:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, http_async_client=client)
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User question: {q1}\n\nDatabase schema:\n{schema}"),
            AIMessage(content='{"plan": {"intent": "Code execution", "steps": []}}'),
            AIMessage(content="Tesla produced the most electric vehicle models in 2022."),
            HumanMessage(content=f"User question: {q2}\n\nDatabase schema:\n{schema}"),
        ]
        resp = await llm.ainvoke(messages)
        content = resp.content.strip("`").removeprefix("json").strip()
        plan_dict = json.loads(content)["plan"]
        plan = Plan.model_validate(plan_dict)

        sql_step = plan.steps[0]
        assert sql_step.type == "SQL_QUERY"
        assert sql_step.semantic_intent is not None
        # Check that Tesla and 2022 were concretely resolved
        desc_and_filters = (sql_step.description + " " + str(sql_step.semantic_intent.filters)).lower()
        assert "tesla" in desc_and_filters
        assert "2022" in desc_and_filters
        assert sql_step.semantic_intent.grain == "entity"
        assert "model_name" in (sql_step.semantic_intent.grouping_columns or [sql_step.semantic_intent.entity_key])

        # Generate SQL
        codegen = CodeGenNode(llm)
        state = {
            "current_question": q2,
            "schema_context": schema,
            "plan": plan,
            "current_step_index": 0,
            "datasets": {},
        }
        result = await codegen(state)
        query = result["generated_code"].upper()
        assert "TESLA" in query
        assert "2022" in query
        assert "GROUP BY" in query
        assert "MODEL_NAME" in query
        assert "SUM(" in query
        assert "LIMIT 3" in query


@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key required for LLM tests")
@pytest.mark.asyncio
async def test_6_explicit_granularity():
    """Test 6 — Explicit granularity: user asks for product variants rather than parent products.

    Question: "What are the top 5 product variants by sales?"
    Expected: Group by variant (or variant column) rather than aggregating to parent product.
    """
    schema = """
{
  "catalog_sales": {
    "columns": [
      {"name": "product_name", "type": "VARCHAR"},
      {"name": "variant_sku", "type": "VARCHAR"},
      {"name": "store_id", "type": "VARCHAR"},
      {"name": "units_sold", "type": "INTEGER"}
    ],
    "sample_rows": [
      {"product_name": "T-Shirt", "variant_sku": "TSHIRT-RED-S", "store_id": "S1", "units_sold": 5},
      {"product_name": "T-Shirt", "variant_sku": "TSHIRT-BLUE-M", "store_id": "S1", "units_sold": 10}
    ]
  }
}
"""
    q = "What are the top 5 product variants by sales?"
    async with httpx.AsyncClient() as client:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, http_async_client=client)
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User question: {q}\n\nDatabase schema:\n{schema}"),
        ]
        resp = await llm.ainvoke(messages)
        content = resp.content.strip("`").removeprefix("json").strip()
        plan_dict = json.loads(content)["plan"]
        plan = Plan.model_validate(plan_dict)

        sql_step = plan.steps[0]
        assert sql_step.type == "SQL_QUERY"
        assert sql_step.semantic_intent is not None
        # Must group by variant_sku, not just product_name
        assert "variant_sku" in (sql_step.semantic_intent.grouping_columns or [sql_step.semantic_intent.entity_key])

        # Generate SQL
        codegen = CodeGenNode(llm)
        state = {
            "current_question": q,
            "schema_context": schema,
            "plan": plan,
            "current_step_index": 0,
            "datasets": {},
        }
        result = await codegen(state)
        query = result["generated_code"].upper()
        assert "VARIANT_SKU" in query
        assert "GROUP BY" in query
        assert "LIMIT 5" in query


@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key required for LLM tests")
@pytest.mark.asyncio
async def test_7_vgsales_regression_q1_and_q2():
    """Test 7 — Full regression of the reported vgsales Q1 & Q2 scenario.

    Q1: "Which publisher released the most titles in the Shooter genre between 2005 and 2010?"
    Q2: "What were their top 3 best-selling Shooter games during that same period?"

    Verifies:
    - Q1 counts distinct titles (COUNT(DISTINCT Name)) grouped by Publisher.
    - Q2 resolves 'their' -> Activision, 'same period' -> 2005-2010.
    - Q2 groups by Name and calculates SUM(Global_Sales) to select top 3.
    - Executing Q2 SQL returns exactly 3 distinct games with no duplicate Black Ops.
    """
    csv_path = Path("data/vgsales.csv")
    with DuckDBEngine(":memory:") as db:
        db.load_dataset(csv_path, "vgsales")
        schema = db.get_schema_summary()

    async with httpx.AsyncClient() as client:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, http_async_client=client)

        # --- Turn 1: Q1 ---
        q1 = "Which publisher released the most titles in the Shooter genre between 2005 and 2010?"
        planner_messages_1 = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"User question: {q1}\n\nDatabase schema:\n{schema}"),
        ]
        resp1 = await llm.ainvoke(planner_messages_1)
        plan1 = Plan.model_validate(json.loads(resp1.content.strip("`").removeprefix("json").strip())["plan"])

        codegen = CodeGenNode(llm)
        state1 = {
            "current_question": q1,
            "schema_context": schema,
            "plan": plan1,
            "current_step_index": 0,
            "datasets": {},
        }
        code_res_1 = await codegen(state1)
        q1_sql = code_res_1["generated_code"]

        # Verify Q1 query properties: counts distinct titles, grouped by Publisher
        assert "COUNT(DISTINCT" in q1_sql.upper() or "COUNT (DISTINCT" in q1_sql.upper()
        assert "PUBLISHER" in q1_sql.upper()

        # Execute Q1 to verify valid results
        with DuckDBEngine(":memory:") as db:
            db.load_dataset(csv_path, "vgsales")
            exec1 = db.execute_read_query(q1_sql)
            assert exec1["success"]
            top_publishers = [r["Publisher"] for r in exec1["rows"][:3]]
            # Electronic Arts (34) and Activision (32) are top publishers
            assert "Activision" in top_publishers or "Electronic Arts" in top_publishers

        # --- Turn 2: Q2 ---
        # In conversation history, Activision was the topic from Q1
        q2 = "What were their top 3 best-selling Shooter games during that same period?"
        planner_messages_2 = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=q1),
            AIMessage(content="Activision released the most titles in the Shooter genre between 2005 and 2010 with 50 titles."),
            HumanMessage(content=f"User question: {q2}\n\nDatabase schema:\n{schema}"),
        ]
        resp2 = await llm.ainvoke(planner_messages_2)
        plan2 = Plan.model_validate(json.loads(resp2.content.strip("`").removeprefix("json").strip())["plan"])

        state2 = {
            "current_question": q2,
            "schema_context": schema,
            "plan": plan2,
            "current_step_index": 0,
            "datasets": {},
        }
        code_res_2 = await codegen(state2)
        q2_sql = code_res_2["generated_code"]

        # Verify Q2 query properties
        assert "ACTIVISION" in q2_sql.upper()
        assert "SHOOTER" in q2_sql.upper()
        assert "GROUP BY" in q2_sql.upper()
        assert "NAME" in q2_sql.upper()
        assert "SUM(" in q2_sql.upper()
        assert "LIMIT 3" in q2_sql.upper()

        # Execute Q2 query and verify results
        with DuckDBEngine(":memory:") as db:
            db.load_dataset(csv_path, "vgsales")
            exec2 = db.execute_read_query(q2_sql)
            assert exec2["success"]
            rows = exec2["rows"]
            assert len(rows) == 3

            game_names = [r["Name"] for r in rows]
            # Verify all 3 game names are distinct (no duplicate Black Ops!)
            assert len(set(game_names)) == 3
            assert "Call of Duty: Black Ops" in game_names
            assert "Call of Duty: Modern Warfare 2" in game_names
            assert "Call of Duty 4: Modern Warfare" in game_names
