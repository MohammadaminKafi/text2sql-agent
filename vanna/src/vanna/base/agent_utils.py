from pydantic import BaseModel, Field


class QueryArgs(BaseModel):
    query: str = Field(..., description="SQL query to be executed")


class AskUserArgs(BaseModel):
    question: str = Field(
        ..., description="Question to ask user for more clarification"
    )


class QueryRAGArgs(BaseModel):
    query: str = Field(
        ..., description="Query to find similarities from vector database"
    )
    count: int = Field(
        ..., description="Number of similar vectors to retreive from vector database"
    )


class ShamsiDateArgs(BaseModel):
    date: str = Field(..., description="Date in Shamsi (Jalali) calendar")
