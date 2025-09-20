from fastapi import FastAPI
from routes import chat

app = FastAPI(
    title="RAG Chatbot API",
    description="Chat with webpages/documents using RAG (LangChain + Unstructured + Pinecone + Gemini).",
    version="1.0.0",
)

# include routes
app.include_router(chat.router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running 🚀"}
