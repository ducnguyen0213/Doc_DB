import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance, UpdateStatus
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import numpy as np

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

class QdrantDocumentDB:
    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int = 768
    ):
        self.qdrant_client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.model = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')

        collections = self.qdrant_client.get_collections().collections
        exists = any(col.name == collection_name for col in collections)
        
        if not exists:
            try:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"Đã tạo collection mới: {collection_name}")
            except Exception as e:
                print(f"Lỗi khi tạo collection: {e}")
        else:
            print(f"Collection {collection_name} đã tồn tại, tiếp tục sử dụng.")

    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def add_document(self, doc_id: str, text: str, metadata: dict):
        embedding = self.get_embedding(text)
        point = PointStruct(
            id=doc_id,
            vector=embedding.tolist(),
            payload=metadata
        )
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        return doc_id

    def update_metadata(self, doc_id: str, metadata: dict):
        self.qdrant_client.set_payload(
            collection_name=self.collection_name,
            payload=metadata,
            points=[doc_id]
        )

    def search_similar(self, query_text: str, top_k: int = 5, min_score: float = 0.0):
        query_vector = self.get_embedding(query_text)
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        filtered_results = [
            {
                "id": hit.id,
                "score": hit.score,
                "metadata": hit.payload
            }
            for hit in results
            if hit.score > min_score
        ]
        return filtered_results

    def get_all_documents(self, offset: int = 0, limit: int = 10):
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        total_docs = collection_info.points_count

        results = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            offset=offset,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        documents = [
            {
                "id": point.id,
                "metadata": point.payload
            }
            for point in results[0]
        ]
        
        return {
            "total": total_docs,
            "items": documents,
            "offset": offset,
            "limit": limit
        } 