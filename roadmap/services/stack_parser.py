import os, json


def detect_stack(repo_path: str):
    stack = []

    # Python
    if os.path.exists(os.path.join(repo_path, "requirements.txt")):
        stack.append("Python")
        with open(os.path.join(repo_path, "requirements.txt")) as f:
            deps = f.read().lower()
            if "fastapi" in deps:
                stack.append("FastAPI")
            if "django" in deps:
                stack.append("Django")

    # Node.js
    if os.path.exists(os.path.join(repo_path, "package.json")):
        stack.append("Node.js")
        with open(os.path.join(repo_path, "package.json")) as f:
            pkg = json.load(f)
            deps = pkg.get("dependencies", {})
            if "react" in deps:
                stack.append("React")
            if "next" in deps:
                stack.append("Next.js")

    # Docker
    if os.path.exists(os.path.join(repo_path, "Dockerfile")):
        stack.append("Docker")

    return list(set(stack))  # unique
