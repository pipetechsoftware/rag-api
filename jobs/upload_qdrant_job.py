from typing import List
from interfaces.qdrant_interface import DocumentInterface, MetadataInterface
from services.qdrant import QdrantService
from settings import QDRANT_COLLECTION
from services.extract import ExtractService


extract = ExtractService()
qdrant_service = QdrantService()


def upload_qdrant_job(job_id: str, agent_id: str, file: bytes) -> None:
    response: List[str] = extract.run(source=file)
    documents: List[DocumentInterface] = [ DocumentInterface(content=doc, metadata=MetadataInterface(index=i, agent_id=agent_id) ) for i, doc in enumerate(response) ]
    qdrant_service.insert_documents(
        collection_name=QDRANT_COLLECTION,
        documents=documents
    )
    # CHAMAR WEBHOOK PARA NOTIFICAR O AGENTE