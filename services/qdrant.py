from typing import  List
from qdrant_client import QdrantClient,models
from sentence_transformers import SentenceTransformer
from interfaces.qdrant_interface import DocumentInterface, ResponseInterface
from settings import  QDRANT_KEY, QDRANT_URL



class QdrantService:
    def __init__(self) -> None:
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
        self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # Modelo otimizado para múltiplos idiomas


    def create_collection(self, collection_name: str) -> None:
        self.client.create_collection(collection_name=collection_name, 
                                      vectors_config={
                                            "text_embedding": models.VectorParams(
                                                size=384,
                                                distance=models.Distance.COSINE
                                            )
                                        }
                                    )
    
    
    def insert_documents(self, collection_name: str, documents: List[DocumentInterface]) -> None:
        documents.sort(key=lambda x: x.metadata.index)
    
        points = [] 

        for doc in documents:
            embedding = self.embedding_model.encode(doc.content).tolist() 
            payload = {
                "index": doc.metadata.index,
                "agent_id": doc.metadata.agent_id,
                "content": doc.content
            }

            points.append(
                models.PointStruct(
                    id=payload["index"],  
                    vector={"text_embedding": embedding},
                    payload=payload  
                )
            )

        self.client.upsert(collection_name=collection_name, points=points)
            

    def query(self, collection_name: str, query: str, agent_id: int, limit: int = 3)->list[ResponseInterface]:
        
        query_embedding = self.embedding_model.encode(query).tolist()  

        search_result = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            using="text_embedding",
            limit=limit,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="agent_id",
                        match=models.MatchValue(value=agent_id)
                    )
                ]
            )
        )
        

        
       

        response: List[ResponseInterface] = [
            ResponseInterface(
                id=str(result.id),
                agent_id=str(result.payload["agent_id"]),
                page_content=result.payload["content"],
                similarity=round(result.score*100, 2)
            )
            for result in search_result.points
        ]
        return response

