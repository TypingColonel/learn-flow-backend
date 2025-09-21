import requests
import os

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def search_google(query: str, max_results: int = 5):
    """
    Uses Tavily Search API to fetch search results.
    """
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {"api_key": TAVILY_API_KEY, "query": query, "num_results": max_results}

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()
    return [
        {"title": r["title"], "url": r["url"], "snippet": r.get("content", "")}
        for r in data.get("results", [])
    ]
