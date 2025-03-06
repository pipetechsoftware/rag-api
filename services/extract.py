from docling.document_converter import DocumentConverter
from typing import List
import tempfile
from langchain_text_splitters import CharacterTextSplitter



class ExtractService:
    def __init__(self) -> None:
        self.converter = DocumentConverter()
        self.text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=150)

    def extract_data(self,  source: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(source)
            tmp.flush()     
            result: str = self.converter.convert(source=tmp.name).document.export_to_markdown()
            return result
    
    def split_text(self, text: str) -> List[str]:
        texts: List[str] = self.text_splitter.split_text(text)
        return texts

    def run(self, source:bytes) -> List[str]:
        result: str = self.extract_data(source)
        response: List[str] = self.split_text(text=result)
        return response
        
        




    
    # async def splitText(self, document:str) -> List[str]:
    #     texts: List[str] = self.text_splitter.split_text(document)
    #     return texts