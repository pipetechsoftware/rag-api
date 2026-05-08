import os
from dotenv import load_dotenv


_HERE = os.path.dirname(os.path.abspath(__file__))
# Suporta `.env` na raiz do serviço ou no diretório pai (monorepo).
load_dotenv(os.path.join(_HERE, ".env"))
load_dotenv(os.path.join(os.path.dirname(_HERE), ".env"))

def _clean_env(value: str) -> str:
    """
    Sanitiza valores vindos do ambiente/.env.
    É comum `.env` ter valores com aspas ('...' / "..."), o que quebra URLs/keys.
    """
    v = (value or "").strip()
    if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ("'", '"')):
        v = v[1:-1].strip()
    return v


QDRANT_URL: str = _clean_env(os.getenv("QDRANT_URL", ""))
QDRANT_KEY: str = _clean_env(os.getenv("QDRANT_KEY", default=""))
QDRANT_COLLECTION: str = _clean_env(os.getenv("QDRANT_COLLECTION", ""))
API_WEBHOOK: str = _clean_env(os.getenv("API_WEBHOOK", ""))

envs: dict[str, str] = {
    "QDRANT_URL": QDRANT_URL,
    "QDRANT_KEY": QDRANT_KEY,
    "QDRANT_COLLECTION": QDRANT_COLLECTION,
    "API_WEBHOOK": API_WEBHOOK
}
for k, v in envs.items():
    if not v:
        raise ValueError(f"Variável de ambiente {k} não definida")