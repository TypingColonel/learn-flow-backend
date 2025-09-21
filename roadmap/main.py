# main.py
from fastapi import FastAPI
from routes import roadmap

app = FastAPI(title="Repo Roadmap Generator")

app.include_router(roadmap.router, prefix="/api")
