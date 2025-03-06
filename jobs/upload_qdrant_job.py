from typing import List

from httpx import request
from interfaces.qdrant_interface import DocumentInterface, MetadataInterface
from services.qdrant import QdrantService
from settings import QDRANT_COLLECTION, API_WEBHOOK
from services.extract import ExtractService


extract = ExtractService()
qdrant_service = QdrantService()


API_URL = API_WEBHOOK+'/api/webhooks/knowledge-base'

def upload_qdrant_job(media_id: str, metadata: str, agent_id: int, file: bytes):

    extracted_docs: List[str] = extract.run(source=file)

    documents: List[DocumentInterface] = [
        DocumentInterface(
            content=doc,
            metadata=MetadataInterface(
                index=i,
                agent_id=agent_id,
                media_id=media_id,
                metadata=metadata
            )
        )
        for i, doc in enumerate(extracted_docs)
    ]

    # Insere os documentos na coleção Qdrant
    response = qdrant_service.insert_documents(
        collection_name=QDRANT_COLLECTION,
        documents=documents
    )

    if not response:
        request("POST", API_URL, json={
            "status": "failed",
            "mediaId": media_id,
            "agentId": agent_id,
            "statusCode": 400
            })
        return False
        
    request("POST", API_URL, json={
        "status": "succeeded",
        "mediaId": media_id,
        "agentId": agent_id,
        "statusCode": 200
    })
    return True
    
