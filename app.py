import uuid

from sympy import Q
from jobs.upload_qdrant_job import upload_qdrant_job
from fastapi import BackgroundTasks, FastAPI, UploadFile, File, Form

from services.extract import ExtractService
from services.qdrant import QdrantService, ResponseInterface

app = FastAPI()
qdrant_service = QdrantService()


@app.post(path="/start-job")
def start_job(
    job_id: str = Form(default=...),
    agent_id: str = Form(default=...),
    file: UploadFile = File(default=...),
    background_tasks: BackgroundTasks = None # type: ignore
):# -> dict[str, Any]:# -> dict[str, Any]:
 
    file_bytes: bytes = file.file.read()
    background_tasks.add_task(upload_qdrant_job, job_id=job_id, agent_id=agent_id, file=file_bytes)
    
    return {
        "status": 200,
        "response": "Estamos processando o arquivo, te notificaremos quando terminar!"
        }

@app.get(path="/search")
def search(
    query: str,
    agent_id: int,
    limit: int = 3,
    ) -> list[ResponseInterface]:
    agent_id = int(agent_id)
    limit = int(limit)
    
    return qdrant_service.query(
        collection_name="teste_19_02",
        query=query, agent_id=agent_id, limit=limit)
    

