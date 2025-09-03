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
    extracted_docs: List[str] = extract.run(source=file)

    documents: List[DocumentInterface] = []
    for i, doc_text in enumerate(extracted_docs):
        documents.append(
            DocumentInterface(
                content=doc_text,
                metadata=MetadataInterface(
                    index=i, agent_id=agent_id, media_id=media_id, metadata=metadata
                ),
            )
        )

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
