# main.py
import uvicorn
from fastapi import FastAPI
from routes import chat

app = FastAPI(title="YouTube Summariser with Gemini + LangGraph")

# Register routes
app.include_router(chat.router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
