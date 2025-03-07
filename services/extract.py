# extract.py

import tempfile
from typing import List

from docling.document_converter import DocumentConverter

# Importamos o splitter específico para Markdown
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class ExtractService:
    def __init__(self) -> None:
        self.converter = DocumentConverter()

        # Define a hierarquia de cabeçalhos que vamos reconhecer.
        # Por exemplo: título (# ), subtítulo (## ), seção (### ), etc.
        self.headers_to_split_on = [
            ("#", "titulo_1"),
            ("##", "titulo_2"),
            ("###", "titulo_3"),
        ]

        # O MarkdownHeaderTextSplitter vai separar o texto em blocos
        # com base nesses headers.
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )

        # Após separar pelos headers, ainda podemos refinar subdividindo blocos grandes
        # com RecursiveCharacterTextSplitter, por exemplo, se ficar maior que chunk_size=1000
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", ".", "?", "!", " ", "\n"],
        )

    def extract_data(self, source: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(source)
            tmp.flush()
            result: str = self.converter.convert(
                source=tmp.name
            ).document.export_to_markdown()
            return result

    def split_text(self, text: str) -> List[str]:
        # Primeiro dividimos em "subdocumentos" por cabeçalhos.
        markdown_splits = self.header_splitter.split_text(text)

        # Cada "subdocumento" pode ainda ser grande, então dividimos recursivamente.
        final_chunks = []
        for md_split in markdown_splits:
            # md_split é do tipo 'Document' (ou algo similar), contendo .page_content
            sub_chunks = self.recursive_splitter.split_text(md_split.page_content)
            final_chunks.extend(sub_chunks)

        return final_chunks

    def run(self, source: bytes) -> List[str]:
        result: str = self.extract_data(source)
        response: List[str] = self.split_text(text=result)
        return response
