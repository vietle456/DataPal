from typing import Literal

from pydantic import BaseModel, Field


class SemanticIntent(BaseModel):
    target_entity: str | None = None
    entity_key: str | None = None
    grain: Literal["entity", "row", "group"] | str | None = None
    grouping_columns: list[str] = Field(default_factory=list)
    measures: list[str] = Field(default_factory=list)
    aggregation_function: str | None = None
    ranking_order: Literal["DESC", "ASC"] | str | None = None
    limit: int | None = None
    filters: list[str] | str | None = None


class PlanStep(BaseModel):
    type: Literal["SQL_QUERY", "PYTHON", "ANSWER"]
    description: str
    semantic_intent: SemanticIntent | None = None


class Plan(BaseModel):
    intent: Literal["Direct answer", "Code execution"]
    steps: list[PlanStep]

