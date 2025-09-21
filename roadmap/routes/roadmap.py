# routes/roadmap.py
from fastapi import APIRouter
import tempfile, subprocess
from services.stack_parser import detect_stack
from services.roadmap_gen import generate_roadmap

router = APIRouter()


@router.post("/roadmap")
def create_roadmap(repo_url: str):
    repo_path = tempfile.mkdtemp()
    subprocess.run(["git", "clone", repo_url, repo_path], check=True)

    stack = detect_stack(repo_path)
    roadmap = generate_roadmap(stack)
    return {"stack": stack, "roadmap": roadmap}
