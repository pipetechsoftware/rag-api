from langchain_qdrant import Qdrant
from docling.document_converter import DocumentConverter
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from fastembed import SparseTextEmbedding

QDRANT_URL: str = 'https://87f833ae-d2cf-4e04-b838-eb649ec8845f.us-east4-0.gcp.cloud.qdrant.io:6333'
QDRANT_KEY: str = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ3MTU5MTkyfQ.8e-ojt9M0AKaW2AQ3iR4ldpyb1JFGlixgOZjWsUBjcI'
OPENAI_API_KEY: str = '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'


QDRANT_CLIENT = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-ada-002",api_key=OPENAI_API_KEY)
DENSE_EMBEDDINGS_MODEL = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
SPARSE_EMBEDDINGS_MODEL = SparseTextEmbedding("Qdrant/bm25")

COLLECTION_NAME = 'noemi'

document_path = 'mais-esperto-que-o-diabo.pdf'

doc_converter = DocumentConverter()
document_text = doc_converter.convert(document_path ).document.export_to_text()

dense_embeddings = DENSE_EMBEDDINGS_MODEL.passage_embed(document_text)
print(len(dense_embeddings))


# def handle(
#     agent_id: str ,
#     document_path: str,
# ):    
    doc_converter = DocumentConverter()
    document_text = doc_converter.convert(document_path ).document.export_to_text()

#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_text(document_text)
    
#     # rececreate collection

    
#     QdrantVectorStore.from_texts(
#         texts=texts,
#         embedding=EMBEDDINGS,
#         url=QDRANT_URL,
#         api_key=QDRANT_KEY,
#         collection_name=COLLECTION_NAME,
#     )

    
# def query(search: str):
#     # Criar embeddings para a consulta
#     search_embedding = EMBEDDINGS.embed_query(search)

#     # Inicializar a coleção no Qdrant com `embedding_function`
#     vector_store = Qdrant(
#         client=QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY),
#         collection_name="mais-esperto-que-o-diabo-02",
#         embeddings=EMBEDDINGS  # O correto é passar isso!
#     )

#     # Fazer a busca por similaridade
#     results = vector_store.similarity_search_by_vector(
#         search_embedding, k=3  # Retorna os 5 resultados mais próximos
#     )

#     return results

# handle(
#     agent_id="agent-01",
#     document_path='mais-esperto-que-o-diabo.pdf')

# # Exemplo de uso
# resultados = query("Como superar o medo?")
# for i, r in enumerate(resultados):
#     print(f"{i+1}. {r.page_content}\n\n")
#     print("---------------------------------------------------"*2)
#     print("\n\n")