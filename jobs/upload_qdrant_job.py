# upload_qdrant_job.py

from typing import List

from httpx import request

from interfaces.qdrant_interface import DocumentInterface, MetadataInterface
from services.extract import ExtractService
from services.qdrant import QdrantService
from settings import API_WEBHOOK, QDRANT_COLLECTION

extract = ExtractService()
qdrant_service = QdrantService()

API_URL = API_WEBHOOK + "/api/webhooks/knowledge-base"


def upload_qdrant_job(media_id: str, metadata: str, agent_id: int, file: bytes):
    # Extrai e chunka o texto
    extracted_docs: List[str] = extract.run(source=file)

    # Monta a lista de DocumentInterface
    documents: List[DocumentInterface] = [
        DocumentInterface(
            content=doc,
            metadata=MetadataInterface(
                index=i, agent_id=agent_id, media_id=media_id, metadata=metadata
            ),
        )
        for i, doc in enumerate(extracted_docs)
    ]

    # Insere no Qdrant
    response = qdrant_service.insert_documents(
        collection_name=QDRANT_COLLECTION, documents=documents
    )

    if not response:
        # Em caso de falha, notifica no webhook
        request(
            "POST",
            API_URL,
            json={
                "status": "failed",
                "mediaId": media_id,
                "agentId": agent_id,
                "statusCode": 400,
            },
        )
        return False

    # Em caso de sucesso, notifica
    # request("POST", API_URL, json={
    #     "status": "succeeded",
    #     "mediaId": media_id,
    #     "agentId": agent_id,
    #     "statusCode": 200
    # })
    return True
