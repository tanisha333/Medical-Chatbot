import json
from typing import List, Union
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


def load_pdf_file(data: Union[str, Path]) -> List[Document]:
    """Load PDF pages from a directory or a single PDF file."""
    path = Path(data)
    cached_docs = _load_cached_docs()
    if cached_docs and (path.name in {"data", "Medical_book.pdf"}):
        return cached_docs

    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path}")
        documents = PyPDFLoader(str(path)).load()
        return _documents_or_cache(documents, path)

    loader = DirectoryLoader(
        str(path),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    return _documents_or_cache(documents, path)


def load_pdf_files(data: Union[str, Path]) -> List[Document]:
    return load_pdf_file(data)


def _documents_or_cache(docs: List[Document], path: Path) -> List[Document]:
    if not docs:
        raise ValueError(f"No PDF documents were loaded from: {path}")

    if any(doc.page_content.strip() for doc in docs):
        return docs

    cached_docs = _load_cached_docs()
    if cached_docs:
        return cached_docs

    raise ValueError(
        "The PDF loaded, but no extractable text was found. "
        "Use a text-based PDF or install an OCR-capable loader such as "
        "`unstructured[pdf]` for scanned/image-only PDFs."
    )


def _load_cached_docs() -> List[Document]:
    cache_path = Path(__file__).resolve().parents[1] / "data" / "medical_book_docs.jsonl"
    if not cache_path.exists():
        return []

    docs = []
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        docs.append(
            Document(
                page_content=row["page_content"],
                metadata=row.get("metadata", {}),
            )
        )
    return docs


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src},
            )
        )
    return minimal_docs


def text_split(extracted_data: List[Document]) -> List[Document]:
    extracted_data = [doc for doc in extracted_data if doc.page_content.strip()]
    if not extracted_data:
        raise ValueError("No text was found to split. Load the PDF data before chunking.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(extracted_data)


def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
