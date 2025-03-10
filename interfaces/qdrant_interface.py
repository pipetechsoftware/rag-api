from pydantic import BaseModel


class MetadataInterface(BaseModel):
    index: int
    agent_id: int
    media_id: str
    metadata: str

class DocumentInterface(BaseModel):
    content: str
    metadata: MetadataInterface

class ResponseInterface(BaseModel):
    id: str
    agent_id: int
    page_content: str
    similarity: float
