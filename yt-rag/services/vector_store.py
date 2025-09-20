# services/vector_store.py
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Env vars
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
existing_indexes = [i["name"] for i in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,  # Gemini embeddings are 768-dim
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
    )

# Get index
index = pc.Index(PINECONE_INDEX_NAME)

# Use Gemini embeddings
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY,
)

# LangChain VectorStore
vectorstore = PineconeVectorStore(index=index, embedding=embedding)


def _chunk_text(text, chunk_size=400, chunk_overlap=250):
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)


def _convert_to_documents(chunks):
    return [Document(page_content=chunk) for chunk in chunks]


def add_text(text):
    chunks = _chunk_text(text)
    documents = _convert_to_documents(chunks)
    vectorstore.add_documents(documents)
    return chunks


def query(question, k=8):
    results = vectorstore.similarity_search(question, k=k)
    return [doc.page_content for doc in results]
