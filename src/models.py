from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Metadata(BaseModel):
    id: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    label: Optional[str] = None
    
    # Cho phép thêm các trường tùy chỉnh khác
    class Config:
        extra = "allow"

class DocumentInput(BaseModel):
    data: str
    metadata: Metadata

class SearchQuery(BaseModel):
    data: str
    top_k: int = 5
    min_score: float = 0.6

class PaginationParams(BaseModel):
    offset: int = 0
    limit: int = 10

class PaginatedResponse(BaseModel):
    total: int
    items: List[Dict[str, Any]]
    offset: int
    limit: int
