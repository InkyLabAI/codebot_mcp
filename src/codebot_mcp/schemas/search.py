from pydantic import BaseModel, Field
from typing import Optional, List, Literal
import uuid
from codebot_mcp.schemas.function import FunctionSearchResult


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    repository_id: Optional[uuid.UUID] = None
    top_k: int = Field(default=10, ge=1, le=1000)
    similarity_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    search_mode: Literal["semantic", "bm25", "hybrid"] = Field(
        default="hybrid",
        description="Search mode: semantic (embeddings), bm25 (keyword), or hybrid (both with RRF)"
    )
    order_by_calls: bool = Field(
        default=False,
        description="Order results by call graph relationships (execution flow)"
    )
    exclude_tests: bool = Field(
        default=True,
        description="Exclude test functions (files in /tests/, test_ prefixed, etc.)"
    )


class ChainResult(BaseModel):
    functions: List[FunctionSearchResult]
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[FunctionSearchResult]  # Flattened results (for backward compatibility)
    total_results: int
    search_mode: str
    ordered_by_calls: bool = False
    call_chain_info: Optional[dict] = None
    chains: Optional[List[ChainResult]] = None  # Structured chains (when ordered_by_calls=True)