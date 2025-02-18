from docling.document_converter import DocumentConverter
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

# Configurações
QDRANT_URL = 'https://87f833ae-d2cf-4e04-b838-eb649ec8845f.us-east4-0.gcp.cloud.qdrant.io:6333'
QDRANT_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ3MTU5MTkyfQ.8e-ojt9M0AKaW2AQ3iR4ldpyb1JFGlixgOZjWsUBjcI'
COLLECTION_NAME = 'noemi_2'
DOCUMENT_PATH = 'mais-esperto-que-o-diabo.pdf'

# Inicializa o QdrantClient e os modelos de embedding
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding("Qdrant/bm25")
late_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

# Converte o PDF para texto
doc_converter = DocumentConverter()
document_text = doc_converter.convert(DOCUMENT_PATH).document.export_to_text()

# Cria a coleção se não existir
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
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

# Chunk do conteúdo usando o CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,    # tamanho máximo de cada chunk
    chunk_overlap=200   # sobreposição entre chunks, se necessário
)
chunks = text_splitter.split_text(document_text)
print(f"Total de chunks: {len(chunks)}")

# Gera os embeddings para cada chunk e prepara os pontos para upload
points = []
for i, chunk in enumerate(chunks, start=1):
    dense_embedding = list(dense_model.passage_embed(chunk))[0]
    sparse_embedding = list(sparse_model.passage_embed(chunk))[0]
    late_embedding = list(late_model.passage_embed(chunk))[0]

    point = models.PointStruct(
        id=i,
        vector={
            "all-MiniLM-L6-v2": dense_embedding.tolist(),
            "bm25": sparse_embedding.as_object(),
            "colbertv2.0": late_embedding.tolist()
        },
        payload={
            "_id": i,
            "title": "mais-esperto-que-o-diabo",
            "text": chunk
        }
    )
    points.append(point)

# Upload dos pontos para o Qdrant
qdrant_client.upload_points(collection_name=COLLECTION_NAME, points=points)
print(f"Total de chunks enviados: {len(points)}")