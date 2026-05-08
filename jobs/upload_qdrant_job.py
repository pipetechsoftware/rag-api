import json
import logging
from typing import List, Optional

from httpx import request

from interfaces.qdrant_interface import DocumentInterface, MetadataInterface
from services.extract import ExtractService
from services.qdrant import MSG_QDRANT_UNREACHABLE, QdrantService
from settings import API_WEBHOOK, QDRANT_COLLECTION

logger = logging.getLogger(__name__)
extract = ExtractService()
qdrant_service = QdrantService()

API_URL = API_WEBHOOK + "/api/webhooks/knowledge-base"


def _maybe_parse_json_from_bytes(file_bytes: bytes) -> Optional[List[dict]]:
    """
    Tenta decodificar bytes como JSON e retorna uma lista de registros, se possível.
    Aceita:
      - Lista de objetos: [ {...}, {...} ]
      - Objeto com QUALQUER chave cujo valor seja uma lista
    """
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("latin-1")
        except UnicodeDecodeError:
            logger.debug("Falha ao decodificar arquivo como texto")
            return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("Falha ao interpretar arquivo como JSON")
        return None

    if isinstance(data, list):
        logger.debug(f"JSON detectado com {len(data)} itens (lista na raiz)")
        return data

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                logger.debug(f"JSON detectado: chave '{key}' contém lista com {len(value)} itens")
                return value

    logger.debug("JSON válido detectado, mas não contém lista para indexação")
    return None


def _build_content_from_item(item: dict) -> str:
    """
    Monta um texto de conteúdo a partir de um item de produto/registro.
    Prioriza campos comuns em PT/EN, fallback para resumo das primeiras chaves.
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
        return content or json.dumps(item, ensure_ascii=False)

    pieces = []
    for k, v in list(item.items())[:8]:
        if isinstance(v, (str, int, float, bool)):
            pieces.append(f"{k}: {v}")
    return " | ".join(pieces) if pieces else json.dumps(item, ensure_ascii=False)


def upload_qdrant_job(media_id: str, metadata: str, agent_id: int, file: bytes):
    """
    Ingestão no Qdrant:
      1. Se for JSON válido → cria documentos direto
      2. Caso contrário → usa Docling para extrair de PDF/DOCX/TXT etc.
    """
    documents: List[DocumentInterface] = []

    try:
        # --- Tenta interpretar como JSON primeiro ---
        json_rows = _maybe_parse_json_from_bytes(file)

        if json_rows is not None:
            logger.debug("Processando como JSON estruturado")
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
                            metadata=json.dumps(item, ensure_ascii=False),
                        ),
                    )
                )

        # --- Caso comum (usa Docling Extract) ---
        else:
            logger.debug("Não detectado JSON, usando Docling para extrair")
            try:
                extracted_docs: List[str] = extract.run(source=file)
            except Exception as e:
                logger.error(f"Docling falhou: {e}")
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
            error_message = response if isinstance(response, str) else "Erro desconhecido ao inserir no Qdrant"
            # Mesma mensagem = cluster/URL (evita ERROR duplicado e stack ruidoso já tratados em services.qdrant)
            if error_message == MSG_QDRANT_UNREACHABLE or (
                isinstance(error_message, str) and error_message.startswith("Não foi possível falar com o Qdrant")
            ):
                logger.warning(
                    "Ingest KB: Qdrant indisponível ou URL/key incorretos | media_id=%s agent_id=%s",
                    media_id,
                    agent_id,
                )
            else:
                logger.error(
                    "Ingest KB: falha ao inserir no Qdrant | media_id=%s agent_id=%s | %s",
                    media_id,
                    agent_id,
                    error_message[:2000],
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
        logger.exception("Erro inesperado durante o processamento do arquivo")
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
