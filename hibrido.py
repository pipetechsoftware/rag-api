from langchain_qdrant import Qdrant
from docling.document_converter import DocumentConverter
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from fastembed import SparseTextEmbedding
from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models

QDRANT_URL: str = 'https://87f833ae-d2cf-4e04-b838-eb649ec8845f.us-east4-0.gcp.cloud.qdrant.io:6333'
QDRANT_KEY: str = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ3MTU5MTkyfQ.8e-ojt9M0AKaW2AQ3iR4ldpyb1JFGlixgOZjWsUBjcI'
# OPENAI_API_KEY: str = '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'
QDRANT_CLIENT = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
# EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-ada-002",api_key=OPENAI_API_KEY)
DENSE_EMBEDDINGS_MODEL = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
SPARSE_EMBEDDINGS_MODEL = SparseTextEmbedding("Qdrant/bm25")
LATE_INTERACTION_MODEL = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")


COLLECTION_NAME = 'noemi'
DOCUMENT_PATH = 'mais-esperto-que-o-diabo.pdf'

# docling

doc_converter = DocumentConverter()
document_text = doc_converter.convert(DOCUMENT_PATH ).document.export_to_text()


QDRANT_CLIENT = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)

if not QDRANT_CLIENT.collection_exists(COLLECTION_NAME):
    QDRANT_CLIENT.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "all-MiniLM-L6-v2": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
            ),
            "colbertv2.0": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                )
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
    )
    
    
dense_embedding = list(DENSE_EMBEDDINGS_MODEL.passage_embed(document_text))[0]
sparse_embedding = list(SPARSE_EMBEDDINGS_MODEL.passage_embed(document_text))[0]
late_interaction_embedding = list(LATE_INTERACTION_MODEL.passage_embed(document_text))[0]



i = 1
QDRANT_CLIENT.upload_points(
    collection_name=COLLECTION_NAME,
    points=[
        models.PointStruct(
            id=i,
            vector={
                "all-MiniLM-L6-v2": dense_embedding.tolist(),       
                "bm25": sparse_embedding.as_object(),
                "colbertv2.0": late_interaction_embedding.tolist()   
            },
            payload={
                "_id": i,
                "title": "mais-esperto-que-o-diabo",
                "text": document_text
            }
        )
    ]
)