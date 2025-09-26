import json
from typing import List, Optional
from httpx import request

from interfaces.qdrant_interface import DocumentInterface, MetadataInterface
from services.extract import ExtractService
from services.qdrant import QdrantService
from settings import API_WEBHOOK, QDRANT_COLLECTION

extract = ExtractService()
qdrant_service = QdrantService()

API_URL = API_WEBHOOK + "/api/webhooks/knowledge-base"


def _maybe_parse_json_from_bytes(file_bytes: bytes) -> Optional[List[dict]]:
    """Tenta decodificar bytes como JSON e retorna uma lista de registros, se possível."""
    text = None
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("latin-1")
        except UnicodeDecodeError:
            return None

    try:
        # tenta decodificar como JSON
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # tenta pegar a primeira chave que contenha lista
        for key, value in data.items():
            if isinstance(value, list):
                return value

    return None


def _build_content_from_item(item: dict) -> str:
    """Monta texto amigável a partir de um item de JSON."""
    name = (
        item.get("nome")
        or item.get("titulo")
        or item.get("name")
        or item.get("title")
        or ""
    )
    desc = (
        item.get("descricao")
        or item.get("descrição")
        or item.get("description")
        or item.get("short_description")
        or item.get("detalhes")
        or ""
    )

    if name or desc:
        content = f"{name} - {desc}".strip(" -")
        return content or json.dumps(item, ensure_ascii=False)

    # fallback: junta algumas chaves/valores
    pieces = []
    for k, v in list(item.items())[:8]:
        if isinstance(v, (str, int, float, bool)):
            pieces.append(f"{k}: {v}")
    return " | ".join(pieces) if pieces else json.dumps(item, ensure_ascii=False)


def upload_qdrant_job(media_id: str, metadata: str, agent_id: int, file: bytes):
    documents: List[DocumentInterface] = []

    try:
        # --- Tenta interpretar como JSON primeiro ---
        json_rows = _maybe_parse_json_from_bytes(file)

        if json_rows is not None:
            print(f"[DEBUG] JSON detectado com {len(json_rows)} registros")
            for i, raw in enumerate(json_rows):
                item = raw if isinstance(raw, dict) else {"value": raw}
                documents.append(
                    DocumentInterface(
                        content=_build_content_from_item(item),
                        metadata=MetadataInterface(
                            index=i,
                            agent_id=agent_id,
                            media_id=media_id,
                            metadata=json.dumps(item, ensure_ascii=False),
                        ),
                    )
                )

        # --- Caso comum (usa Docling Extract) ---
        else:
            print("[DEBUG] Não detectado JSON, usando Docling para extrair")
            try:
                extracted_docs: List[str] = extract.run(source=file)
            except Exception as e:
                print(f"[ERROR] Docling falhou: {e}")
                request(
                    "POST",
                    API_URL,
                    json={
                        "status": "failed",
                        "mediaId": media_id,
                        "agentId": agent_id,
                        "statusCode": 415,
                        "message": f"Arquivo não suportado pelo Docling: {str(e)}",
                    },
                )
                return False

            for i, doc_text in enumerate(extracted_docs):
                documents.append(
                    DocumentInterface(
                        content=doc_text,
                        metadata=MetadataInterface(
                            index=i,
                            agent_id=agent_id,
                            media_id=media_id,
                            metadata=metadata,
                        ),
                    )
                )

        # --- Insere no Qdrant ---
        response = qdrant_service.insert_documents(
            collection_name=QDRANT_COLLECTION, documents=documents
        )

        if response is not True:
            error_message = (
                response if isinstance(response, str) else "Erro desconhecido ao inserir no Qdrant"
            )
            request(
                "POST",
                API_URL,
                json={
                    "status": "failed",
                    "mediaId": media_id,
                    "agentId": agent_id,
                    "statusCode": 400,
                    "message": error_message,
                },
            )
            return False

        request(
            "POST",
            API_URL,
            json={
                "status": "succeeded",
                "mediaId": media_id,
                "agentId": agent_id,
                "statusCode": 200,
            },
        )
        return True

    except Exception as e:
        print(f"[ERROR] Erro inesperado: {e}")
        request(
            "POST",
            API_URL,
            json={
                "status": "failed",
                "mediaId": media_id,
                "agentId": agent_id,
                "statusCode": 500,
                "message": str(e),
            },
        )
        return False
