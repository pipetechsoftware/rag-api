import json
from typing import List

from httpx import request

from interfaces.qdrant_interface import DocumentInterface, MetadataInterface
from services.extract import ExtractService
from services.qdrant import QdrantService
from settings import API_WEBHOOK, QDRANT_COLLECTION

extract = ExtractService()
qdrant_service = QdrantService()

API_URL = API_WEBHOOK + "/api/webhooks/knowledge-base"


def upload_qdrant_job(media_id: str, metadata: str, agent_id: int, file: str):
    """
    Faz upload de documentos para o Qdrant.
    Suporta arquivos comuns (PDF/DOCX/TXT) e JSON.
    """

    documents: List[DocumentInterface] = []

    try:
        # --- Caso seja JSON ---
        if file.endswith(".json") or file.endswith(".json.txt"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Garante que seja lista
            if not isinstance(data, list):
                raise ValueError("O JSON precisa ser uma lista de objetos")

            for i, item in enumerate(data):
                # Personalize o conteúdo do chunk
                content = f"{item.get('nome', '')} - {item.get('descricao', '')}"

                documents.append(
                    DocumentInterface(
                        content=content.strip(),
                        metadata=MetadataInterface(
                            index=i,
                            agent_id=agent_id,
                            media_id=media_id,
                            metadata=item,  # guarda todo o objeto como metadata
                        ),
                    )
                )

        # --- Caso comum (usa Docling Extract) ---
        else:
            extracted_docs: List[str] = extract.run(source=file)
            for i, doc_text in enumerate(extracted_docs):
                documents.append(
                    DocumentInterface(
                        content=doc_text,
                        metadata=MetadataInterface(
                            index=i,
                            agent_id=agent_id,
                            media_id=media_id,
                            metadata=metadata,
                        ),
                    )
                )

        # --- Inserir no Qdrant ---
        response = qdrant_service.insert_documents(
            collection_name=QDRANT_COLLECTION, documents=documents
        )

        if response is not True:
            error_message = response if isinstance(response, str) else "Erro desconhecido"
            request(
                "POST",
                API_URL,
                json={
                    "status": "failed",
                    "mediaId": media_id,
                    "agentId": agent_id,
                    "statusCode": 400,
                    "message": error_message,
                },
            )
            return False

        request(
            "POST",
            API_URL,
            json={
                "status": "succeeded",
                "mediaId": media_id,
                "agentId": agent_id,
                "statusCode": 200,
            },
        )
        return True

    except Exception as e:
        request(
            "POST",
            API_URL,
            json={
                "status": "failed",
                "mediaId": media_id,
                "agentId": agent_id,
                "statusCode": 500,
                "message": str(e),
            },
        )
        return False
