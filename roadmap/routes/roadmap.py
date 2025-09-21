from fastapi import APIRouter
import tempfile, subprocess
from services.stack_parser import detect_stack
from services.roadmap_gen import generate_roadmap
from tools.youtube_tool import search_youtube
from tools.web_search import search_google
from tools.github_tools import search_github

router = APIRouter()


@router.post("/roadmap")
def create_roadmap(repo_url: str):
    repo_path = tempfile.mkdtemp()
    subprocess.run(["git", "clone", repo_url, repo_path], check=True)

    stack = detect_stack(repo_path)
    roadmap = generate_roadmap(stack)
    return {"stack": stack, "roadmap": roadmap}

@router.get("/tools/youtube")
def youtube_tool(q: str):
    return search_youtube(q)


@router.get("/tools/google")
def google_tool(q: str):
    return search_google(q)


@router.get("/tools/github")
def github_tool(q: str):
    return search_github(q)