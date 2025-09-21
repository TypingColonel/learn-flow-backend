import requests


def search_github(query: str, max_results=5):
    url = "https://api.github.com/search/repositories"
    params = {"q": query, "per_page": max_results, "sort": "stars"}
    resp = requests.get(url, params=params).json()
    return [
        {
            "name": r["name"],
            "url": r["html_url"],
            "stars": r["stargazers_count"],
            "description": r["description"],
        }
        for r in resp.get("items", [])
    ]
