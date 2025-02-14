
DOCUMENT_PATH: str = 'mais-esperto-que-o-diabo.pdf'
QDRANT_URL: str = 'https://87f833ae-d2cf-4e04-b838-eb649ec8845f.us-east4-0.gcp.cloud.qdrant.io:6333'
PINECONE_KEY: str = "pcsk_6B7DLW_HFydc71QXtMccAcS7zHyf8DuSRqu2D32zSSdYZGM5wrUuAudng7Q43M8wQoK74K"
OPENAI_API_KEY: str = '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'

from docling.document_converter import DocumentConverter
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-ada-002",api_key=OPENAI_API_KEY)
