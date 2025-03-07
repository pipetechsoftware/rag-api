import tempfile
from typing import List

from docling.document_converter import DocumentConverter
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class ExtractService:
    def __init__(self) -> None:
        self.converter = DocumentConverter()

        self.headers_to_split_on = [
            ("#", "titulo_1"),
            ("##", "titulo_2"),
            ("###", "titulo_3"),
        ]

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )

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

        markdown_splits = self.header_splitter.split_text(text)

        final_chunks = []
        for md_split in markdown_splits:

            sub_chunks = self.recursive_splitter.split_text(md_split.page_content)
            final_chunks.extend(sub_chunks)

        return final_chunks

    def run(self, source: bytes) -> List[str]:
        result: str = self.extract_data(source)
        response: List[str] = self.split_text(text=result)
        return response
