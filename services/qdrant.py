import hashlib
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http import models as http_models
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
        """
        Cria (ou recria) a coleção sem usar payload_schema,
        pois Qdrant 1.x não suporta text indexing nativo.
        """

        collections = self.client.get_collections().collections
        existing = [c.name for c in collections if c.name == collection_name]
        if existing:
            self.client.delete_collection(collection_name=collection_name)

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
        Insere documentos, gerando embeddings para cada chunk e
        deduplicando dentro da mesma chamada (via hash do conteúdo).
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
                    "content": doc.content,
                    "agent_id": doc.metadata.agent_id,
                    "media_id": doc.metadata.media_id,
                    "metadata": doc.metadata.metadata,
                    "index": doc.metadata.index,
                }

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={"text_embedding": embedding},
                        payload=payload,
                    )
                )

            self.client.upsert(collection_name=collection_name, points=points)
            return True

        except Exception as e:
            return str(e)

    def query(
        self,
        collection_name: str,
        query: str,
        agent_id: Optional[int] = None,
        media_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[ResponseInterface]:
        """
        Implementa uma 'busca híbrida manual':
          1. Busca vetorial (com query_points).
          2. Busca lexical (substring) local - puxa top N docs e filtra
             ou puxa todos, dependendo do volume.
          3. Combina resultados, priorizando docs que apareçam nas duas listas.
        """

        query_embedding = self.embedding_model.encode(query).tolist()

        must_conditions = []
        if agent_id is not None:
            must_conditions.append(
                http_models.FieldCondition(
                    key="agent_id", match=http_models.MatchValue(value=agent_id)
                )
            )
        if media_id is not None:
            must_conditions.append(
                http_models.FieldCondition(
                    key="media_id", match=http_models.MatchValue(value=media_id)
                )
            )

        vector_search = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            using="text_embedding",
            limit=limit,
            query_filter=(
                http_models.Filter(must=must_conditions) if must_conditions else None
            ),
        )

        vector_hits: List[ResponseInterface] = [
            ResponseInterface(
                id=str(result.id),
                agent_id=str(result.payload.get("agent_id", None)),  # type: ignore
                page_content=result.payload.get("content", ""),  # type: ignore
                similarity=result.score,
            )
            for result in vector_search.points
        ]

        lexical_limit = max(limit, 100)
        lexical_search = self.client.query_points(
            collection_name=collection_name,
            query=[0.0] * 768,
            using="text_embedding",
            limit=lexical_limit,
            query_filter=(
                http_models.Filter(must=must_conditions) if must_conditions else None
            ),
        )

        query_lower = query.lower()
        lexical_matches = []
        for point in lexical_search.points:
            content = point.payload["content"].lower()  # type: ignore
            if query_lower in content:

                lexical_matches.append(
                    ResponseInterface(
                        id=str(point.id),
                        agent_id=str(point.payload.get("agent_id", None)),  # type: ignore
                        page_content=point.payload.get("content", ""),  # type: ignore
                        similarity=0.99,
                    )
                )

        fused = {}
        for vh in vector_hits:
            fused[vh.id] = vh

        for lm in lexical_matches:
            if lm.id in fused:

                if lm.similarity > fused[lm.id].similarity:
                    fused[lm.id].similarity = lm.similarity
            else:

                fused[lm.id] = lm

        all_results = list(fused.values())
        all_results.sort(key=lambda x: x.similarity, reverse=True)
        final_results = all_results[:limit]

        return final_results

    def delete_vectors(
        self,
        collection_name: str,
        agent_id: Optional[int] = None,
        media_id: Optional[str] = None,
    ) -> bool:
        """
        Deleta todos os vetores que tenham o agent_id e/ou media_id.
        Verifica se existe algo antes de deletar.
        """
        must_conditions = []
        if agent_id is not None:
            must_conditions.append(
                http_models.FieldCondition(
                    key="agent_id", match=http_models.MatchValue(value=agent_id)
                )
            )
        if media_id is not None:
            must_conditions.append(
                http_models.FieldCondition(
                    key="media_id", match=http_models.MatchValue(value=media_id)
                )
            )

        if not must_conditions:
            raise ValueError("Informe ao menos um filtro: agent_id ou media_id.")

        check = self.client.query_points(
            collection_name=collection_name,
            query=[0.0] * 768,
            using="text_embedding",
            limit=1,
            query_filter=http_models.Filter(must=must_conditions),
        )

        if not check.points:
            return False

        self.client.delete(
            collection_name=collection_name,
            points_selector=http_models.Filter(must=must_conditions),
        )
        return True
