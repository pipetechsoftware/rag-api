import hashlib
import uuid
import warnings
from typing import List, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http import models as http_models
from sentence_transformers import SentenceTransformer

from interfaces.qdrant_interface import DocumentInterface, ResponseInterface
from settings import QDRANT_KEY, QDRANT_URL


def _normalize_qdrant_url(raw: str) -> str:
    """
    Aceita:
      - https://host:port
      - http://host:port
      - host:port (comum em painéis de env)
    """
    url = (raw or "").strip()
    if not url:
        return url
    if "://" not in url:
        # Qdrant Cloud normalmente exige HTTPS.
        url = f"https://{url}"
    return url


class QdrantService:
    def __init__(self) -> None:

        qdrant_url = _normalize_qdrant_url(QDRANT_URL)
        # Algumas versões do `qdrant-client` não aceitam `check_version` no construtor.
        # Fazemos fallback para manter compatibilidade.
        try:
            self.client = QdrantClient(
                url=qdrant_url, api_key=QDRANT_KEY, check_version=False
            )
        except TypeError:
            # Algumas versões não aceitam `check_version` (ou repassam o kw para o httpx e quebram).
            # O client ainda funciona; só não consulta a versão do servidor na inicialização.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Failed to obtain server version.*",
                    category=UserWarning,
                )
                self.client = QdrantClient(url=qdrant_url, api_key=QDRANT_KEY)

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2", device="cpu"
        )

    def create_collection(self, collection_name: str) -> None:
        """
        Cria (ou recria) a coleção e adiciona índices de payload
        para permitir filtros por agent_id e media_id.
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

        # 🔑 Criar índice para filtros
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="agent_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="media_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
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
                    key="agent_id",
                    match=http_models.MatchValue(value=int(agent_id))
                )
            )
        if media_id is not None:
            must_conditions.append(
                http_models.FieldCondition(
                    key="media_id",
                    match=http_models.MatchValue(value=str(media_id))
                )
            )

        try:
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
                    agent_id=str(result.payload.get("agent_id", None)),
                    page_content=result.payload.get("content", ""),
                    similarity=result.score,
                )
                for result in vector_search.points
            ]
        except Exception as e:
            print(f"[QdrantService.query] Erro na busca vetorial: {e}")
            vector_hits = []

        # 🔹 Busca lexical (string match local)
        lexical_limit = max(limit, 100)
        try:
            lexical_search = self.client.query_points(
                collection_name=collection_name,
                query=[0.0] * 768,  # vetor "dummy"
                using="text_embedding",
                limit=lexical_limit,
                query_filter=(
                    http_models.Filter(must=must_conditions) if must_conditions else None
                ),
            )
        except Exception as e:
            print(f"[QdrantService.query] Erro na busca lexical: {e}")
            lexical_search = http_models.QueryResponse(points=[])

        query_lower = query.lower()
        lexical_matches = []
        for point in lexical_search.points:
            content = point.payload.get("content", "").lower()  # type: ignore
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
                fused[lm.id].similarity = max(lm.similarity, fused[lm.id].similarity)
            else:
                fused[lm.id] = lm

        # Ordena por similaridade e limita
        all_results = list(fused.values())
        all_results.sort(key=lambda x: x.similarity, reverse=True)
        return all_results[:limit]

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
