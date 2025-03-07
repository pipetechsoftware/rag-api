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

        # Exemplo de modelo de embedding (768 dimensões):
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
            # Ordenamos os documentos pelo index para manter consistência
            documents.sort(key=lambda x: x.metadata.index)

            # Set para guardar hashes já vistos (evitando inserir chunks duplicados)
            seen_hashes = set()
            points = []

            for doc in documents:
                # Normalizamos o texto (opcional) para reduzir chances de duplicação
                normalized_content = doc.content.strip().lower()

                # Gera hash (ex.: SHA-256) do conteúdo
                content_hash = hashlib.sha256(
                    normalized_content.encode("utf-8")
                ).hexdigest()

                # Se esse hash já estiver no set, significa que esse chunk é igual a outro
                if content_hash in seen_hashes:
                    # pula este chunk
                    continue

                # Se não estiver, adiciona ao set e continua
                seen_hashes.add(content_hash)

                # Cria embedding
                embedding = self.embedding_model.encode(doc.content).tolist()

                # Gera ID único (UUID4)
                point_id = str(uuid.uuid4())

                payload = {
                    "index": doc.metadata.index,
                    "agent_id": doc.metadata.agent_id,
                    "content": doc.content,
                    "media_id": doc.metadata.media_id,
                    "metadata": doc.metadata.metadata,
                }

                # Monta o point e adiciona à lista
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={"text_embedding": embedding},
                        payload=payload,
                    )
                )

            # Se não tiver nada para inserir (por exemplo, tudo era duplicado), ainda
            # é "True", mas não vai inserir nada
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
        # Gera o embedding da query
        query_embedding = self.embedding_model.encode(query).tolist()

        # Monta o filtro condicional
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
                agent_id=str(result.payload["agent_id"]),  # type: ignore
                page_content=result.payload["content"],  # type: ignore
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

        # Verifica existência
        search_result = self.client.query_points(
            collection_name=collection_name,
            query=[0.0] * 768,  # vetor "nulo" ou placeholder
            using="text_embedding",
            limit=1,
            query_filter=models.Filter(must=filters),
        )

        if not search_result.points:
            return False

        # Se existe algo que bate o filtro, deletamos
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.Filter(must=filters),
        )
        return True
