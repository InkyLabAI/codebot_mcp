from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid


class FunctionBase(BaseModel):
    function_id: str
    name: str
    file_path: str
    class_name: Optional[str] = None
    nested: Optional[str] = None
    code: str
    docstring: Optional[str] = None
    start_line: int
    end_line: int


class FunctionCreate(FunctionBase):
    repository_id: uuid.UUID
    parameters: Optional[List[Dict[str, Any]]] = None
    decorators: Optional[List[str]] = None
    return_type: Optional[str] = None
    calls: Optional[List[str]] = None


class FunctionResponse(FunctionBase):
    id: uuid.UUID
    repository_id: uuid.UUID
    parameters: Optional[List[Dict[str, Any]]] = None
    decorators: Optional[List[str]] = None
    return_type: Optional[str] = None
    calls: Optional[List[str]] = None  # Added for call graph ordering
    module_docstring: Optional[str] = None
    class_docstring: Optional[str] = None
    created_at: datetime
    has_embedding: bool = False
    
    model_config = ConfigDict(from_attributes=True)


class FunctionSearchResult(BaseModel):
    """Search result with similarity score"""
    function: FunctionResponse
    similarity: float
    repository_name: str

    # RRF-specific fields (for hybrid search)
    rrf_score: Optional[float] = None
    semantic_rank: Optional[int] = None
    bm25_rank: Optional[int] = None

    # PCST bridge node indicator
    is_bridge: bool = False