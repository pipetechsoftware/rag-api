from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile

from jobs.upload_qdrant_job import upload_qdrant_job
from services.qdrant import QdrantService, ResponseInterface
from settings import QDRANT_COLLECTION

app = FastAPI()
qdrant_service = QdrantService()


@app.post(path="/create-collection")
def create_collection(collection_name: str = Form(...)):

    return qdrant_service.create_collection(collection_name=collection_name)


@app.post(path="/start-job")
def start_job(
    media_id: str = Form(...),
    metadata: str = Form(...),
    agent_id: int = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    file_bytes: bytes = file.file.read()
    agent_id = int(agent_id)

    background_tasks.add_task(
        upload_qdrant_job,
        media_id=media_id,
        metadata=metadata,
        agent_id=agent_id,
        file=file_bytes,
    )

    return {
        "status": 200,
        "response": "Estamos processando o arquivo. Notificaremos quando terminar!",
    }


@app.get(path="/search")
def search(
    query: str,
    agent_id: Optional[int] = None,
    media_id: Optional[str] = None,
    limit: int = 5,
) -> list[ResponseInterface]:
    if agent_id:
        agent_id = int(agent_id)
    limit = int(limit)

    return qdrant_service.query(
        collection_name=QDRANT_COLLECTION,
        query=query,
        agent_id=agent_id,
        limit=limit,
        media_id=media_id,
    )


@app.delete(path="/delete")
def delete_vectors(
    agent_id: Optional[int] = None,
    media_id: Optional[str] = None,
):
    if (agent_id is None) and (media_id is None):
        raise HTTPException(
            status_code=400, detail="Informe ao menos um filtro: agent_id ou media_id."
        )

    deleted = qdrant_service.delete_vectors(
        collection_name=QDRANT_COLLECTION, agent_id=agent_id, media_id=media_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404, detail="Não existem dados correspondentes na vector store."
        )

    return {"status": 200, "response": "Vetores deletados com sucesso."}
