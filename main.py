from fastapi import FastAPI, HTTPException, Query
import uuid
from src.models import DocumentInput, Metadata, SearchQuery, PaginationParams, PaginatedResponse
from src.qdrant_db import QdrantDocumentDB

app = FastAPI()

# Khởi tạo QdrantDB
db = QdrantDocumentDB(
    url="http://localhost:6333",
    collection_name="documents"
)

@app.post("/upload")
async def upload_document(document: DocumentInput):
    try:
        doc_id = document.metadata.id or str(uuid.uuid4())

        document.metadata.id = doc_id
        
        db.add_document(
            doc_id=doc_id,
            text=document.data,
            metadata=document.metadata.dict()
        )
        return {"status": "success", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/update_metadata/{doc_id}")
async def update_metadata(doc_id: str, metadata: Metadata):
    try:
        # Đảm bảo giữ nguyên id
        metadata.id = doc_id
        db.update_metadata(doc_id=doc_id, metadata=metadata.dict())
        return {"status": "success", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_documents(query: SearchQuery):
    try:
        results = db.search_similar(
            query_text=query.data,
            top_k=query.top_k,
            min_score=query.min_score
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=PaginatedResponse)
async def get_all_documents(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=10, ge=1, le=100)
):
    try:
        return db.get_all_documents(offset=offset, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
