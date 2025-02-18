from langchain_text_splitters import CharacterTextSplitter
from qdrant_client import QdrantClient,models
from docling.document_converter import DocumentConverter
from langchain_community.vectorstores import Qdrant as QdrantVectorStore

from fastembed import TextEmbedding

QDRANT_URL: str = 'https://87f833ae-d2cf-4e04-b838-eb649ec8845f.us-east4-0.gcp.cloud.qdrant.io:6333'
QDRANT_KEY: str = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ3MTU5MTkyfQ.8e-ojt9M0AKaW2AQ3iR4ldpyb1JFGlixgOZjWsUBjcI'
SOURCE = 'mais-esperto-que-o-diabo.pdf'
COLLECTION_NAME = 'noemi'
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'



client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)


def extractData(source:str):
    converter = DocumentConverter()
    document = converter.convert(source=source).document.export_to_markdown()
    return document

def splitText(document:str, chunk_size:int=1000, chunk_overlap:int=200) -> models.List[str]:
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts: models.List[str] = text_splitter.split_text(document)
    return texts


def upload(agent_id: str, source: str):
    document = extractData(source=source)
    chunks = splitText(document=document)

    embedding = TextEmbedding(model_name=EMBEDDING_MODEL)
    
    
    for idx, chunk in enumerate(iterable=chunks):
        # vector = embedding.passage_embed(texts=chunk)
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=idx,
                    vector=models.VectorStruct( # type: ignore
                        text=chunk,
                    ),
                    payload={
                        "agent_id": agent_id,
                    }
                    
                )
            ]
        )
        
        
        return
    
    
    

def create_collection():
    
    if client.collection_exists(COLLECTION_NAME):
        return
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "embedding": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        }
    )


def main(file: str, agent_id: str) -> dict:
    # create_collection()
    # upload(agent_id=agent_id, source=file)
    return {
        "status": 200,
        "message": "ok"
    }

if __name__ == '__main__':
    upload(agent_id='1', source=SOURCE)
    # create_collection()
    
    
                             