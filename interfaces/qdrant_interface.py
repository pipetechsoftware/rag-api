from pydantic import BaseModel


class MetadataInterface(BaseModel):
    index: int
    agent_id: str

class DocumentInterface(BaseModel):
    content: str
    metadata: MetadataInterface

class ResponseInterface(BaseModel):
    id: str
    agent_id: str
    page_content: str
    similarity: float
