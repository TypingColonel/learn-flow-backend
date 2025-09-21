from groq import Groq
import json, os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)


def generate_roadmap(tech_stack: list):
    prompt = f"""
    You are an expert coding mentor.
    A student wants to learn how to build a project using: {tech_stack}.

    Generate a step-by-step learning roadmap in JSON format.
    Each node must include:
      - title
      - summary
      - resources (official docs, GitHub repos, YouTube tutorials)
      - prerequisites (list of earlier nodes)
      - estimated_time
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )

    # FIXED: use .content instead of ["content"]
    text = response.choices[0].message.content

    try:
        return json.loads(text)  # expect JSON
    except:
        return {"error": "Invalid JSON from LLM", "raw": text}
