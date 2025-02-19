import os
from dotenv import load_dotenv



load_dotenv()


QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_KEY: str = os.getenv("QDRANT_KEY", default="")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "")


envs: dict[str, str] = {
    "QDRANT_URL": QDRANT_URL,
    "QDRANT_KEY": QDRANT_KEY,
    "QDRANT_COLLECTION": QDRANT_COLLECTION
}
for k, v in envs.items():
    if not v:
        raise ValueError(f"Variável de ambiente {k} não definida")