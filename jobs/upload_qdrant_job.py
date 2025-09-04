import json
from typing import List, Optional, Union

from httpx import request

from interfaces.qdrant_interface import DocumentInterface, MetadataInterface
from services.extract import ExtractService
from services.qdrant import QdrantService
from settings import API_WEBHOOK, QDRANT_COLLECTION

extract = ExtractService()
qdrant_service = QdrantService()

API_URL = API_WEBHOOK + "/api/webhooks/knowledge-base"


def _maybe_parse_json_from_bytes(file_bytes: bytes) -> Optional[List[dict]]:
    """
    Tenta decodificar bytes como UTF-8 e carregar JSON.
    Retorna uma lista de dicts (registros) ou None se não parecer JSON.
    Aceita formatos:
      - Lista de objetos: [ {...}, {...} ]
      - Objeto com lista dentro: { "items": [ ... ] } (ou "data", "products", etc.)
    """
    try:
        text = file_bytes.decode("utf-8-sig")  # lida com BOM
    except UnicodeDecodeError:
        return None

    s = text.strip()
    if not s or s[0] not in "[{":
        return None

    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return None

    if isinstance(data, list):
        # Lista de registros
        return data
    if isinstance(data, dict):
        # Tenta encontrar uma lista em chaves comuns
        for key in ("items", "data", "products", "records", "result"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    return None


def _build_content_from_item(item: dict) -> str:
    """
    Monta um texto de conteúdo a partir de um item de produto/registro.
    Tenta campos comuns em PT/EN. Se não achar, cria um resumo dos primeiros pares chave:valor.
    """
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
        if not content:
            content = json.dumps(item, ensure_ascii=False)
        return content

    # fallback: sumariza alguns campos simples
    pieces = []
    for k, v in list(item.items())[:8]:
        if isinstance(v, (str, int, float, bool)) and k not in ("id_interno",):
            pieces.append(f"{k}: {v}")
    if pieces:
        return " | ".join(pieces)
    return json.dumps(item, ensure_ascii=False)


def upload_qdrant_job(media_id: str, metadata: str, agent_id: int, file: bytes):
    """
    Ingestão no Qdrant:
      - Se o conteúdo for JSON válido (bytes → utf-8 → json.loads), cria documentos direto.
      - Caso contrário, usa o extrator (Docling) para PDF/DOCX/TXT etc.
    """
    documents: List[DocumentInterface] = []

    try:
        # Tenta JSON primeiro (funciona para .json e .txt)
        json_rows = _maybe_parse_json_from_bytes(file)

        if json_rows is not None:
            # JSON reconhecido
            for i, raw in enumerate(json_rows):
                item = raw if isinstance(raw, dict) else {"value": raw}
                content = _build_content_from_item(item)
                documents.append(
                    DocumentInterface(
                        content=content,
                        metadata=MetadataInterface(
                            index=i,
                            agent_id=agent_id,
                            media_id=media_id,
                            # Mantém como string (Pydantic espera str):
                            metadata=json.dumps(item, ensure_ascii=False),
                        ),
                    )
                )

        # --- Caso comum (usa Docling Extract) ---
        else:
            # Não era JSON → usa o pipeline Docling normalmente
            extracted_docs: List[str] = extract.run(source=file)
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

        # Inserir no Qdrant
        response = qdrant_service.insert_documents(
            collection_name=QDRANT_COLLECTION, documents=documents
        )

        if response is not True:
            error_message = response if isinstance(response, str) else "Erro desconhecido"
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
