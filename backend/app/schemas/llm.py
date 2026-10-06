from pydantic import BaseModel


class LLMResponse(BaseModel):
    """
    Response model for the LLM.
    """

    answer: str
