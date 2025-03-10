import hashlib
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from interfaces.qdrant_interface import DocumentInterface, ResponseInterface
from settings import QDRANT_KEY, QDRANT_URL


class QdrantService:
    def __init__(self) -> None:
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2", device="cpu"
        )

    def create_collection(self, collection_name: str) -> None:
        """Cria a coleção Qdrant com tamanho de embedding = 768 (mpnet-base-v2)."""
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "text_embedding": models.VectorParams(
                    size=768, distance=models.Distance.COSINE
                )
            },
        )

    def insert_documents(
        self, collection_name: str, documents: List[DocumentInterface]
    ) -> bool:
        """
        Insere documentos na collection, gerando embedding para cada chunk.
        Agora com deduplicação de chunks durante a mesma chamada.
        """
        try:

            documents.sort(key=lambda x: x.metadata.index)

            seen_hashes = set()
            points = []

            for doc in documents:

                normalized_content = doc.content.strip().lower()

                content_hash = hashlib.sha256(
                    normalized_content.encode("utf-8")
                ).hexdigest()

                if content_hash in seen_hashes:

                    continue

                seen_hashes.add(content_hash)

                embedding = self.embedding_model.encode(doc.content).tolist()

                point_id = str(uuid.uuid4())

                payload = {
                    "index": doc.metadata.index,
                    "agent_id": doc.metadata.agent_id,
                    "content": doc.content,
                    "media_id": doc.metadata.media_id,
                    "metadata": doc.metadata.metadata,
                }

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={"text_embedding": embedding},
                        payload=payload,
                    )
                )

            if not points:
                print("Nenhum chunk novo para inserir (ou tudo era duplicado).")

            self.client.upsert(collection_name=collection_name, points=points)
            return True

        except Exception as e:
            print("Erro ao inserir documentos no Qdrant:", e)
            return False

    def query(
        self,
        collection_name: str,
        query: str,
        agent_id: Optional[int] = None,
        media_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[ResponseInterface]:

        query_embedding = self.embedding_model.encode(query).tolist()

        must = []
        if agent_id is not None:
            must.append(
                models.FieldCondition(
                    key="agent_id", match=models.MatchValue(value=agent_id)
                )
            )
        if media_id is not None:
            must.append(
                models.FieldCondition(
                    key="media_id", match=models.MatchValue(value=media_id)
                )
            )

        search_result = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            using="text_embedding",
            limit=limit,
            query_filter=models.Filter(must=must),
        )

        response: List[ResponseInterface] = [
            ResponseInterface(
                id=str(result.id),
                agent_id=str(result.payload["agent_id"]),
                page_content=result.payload["content"],
                similarity=result.score,
            )
            for result in search_result.points
        ]

        if response:
            similarity_example = round(response[0].similarity * 100, 2)
            print(f"Primeiro resultado - Similaridade: {similarity_example}%")

        return response

    def delete_vectors(
        self,
        collection_name: str,
        agent_id: Optional[int] = None,
        media_id: Optional[str] = None,
    ) -> bool:
        """
        Deleta todos os vetores que correspondam ao agent_id ou media_id informado.
        Aqui usamos query_points para verificar a existência antes de deletar.
        """
        filters = []
        if agent_id is not None:
            filters.append(
                models.FieldCondition(
                    key="agent_id", match=models.MatchValue(value=agent_id)
                )
            )
        if media_id is not None:
            filters.append(
                models.FieldCondition(
                    key="media_id", match=models.MatchValue(value=media_id)
                )
            )
        if not filters:
            raise ValueError("Informe ao menos um filtro: agent_id ou media_id.")

        search_result = self.client.query_points(
            collection_name=collection_name,
            query=[0.0] * 768,
            using="text_embedding",
            limit=1,
            query_filter=models.Filter(must=filters),
        )

        if not search_result.points:
            return False

        self.client.delete(
            collection_name=collection_name,
            points_selector=models.Filter(must=filters),
        )
        return True
