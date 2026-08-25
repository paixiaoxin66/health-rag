from langchain_text_splitters import RecursiveCharacterTextSplitter

from health_rag.config.settings import get_settings


def split_documents(documents):
    """Split documents into chunks using application settings."""
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            " ",
            "",
        ],
    )

    return splitter.split_documents(documents)