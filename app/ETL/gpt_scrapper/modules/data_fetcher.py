import requests

def fetch_data(keyword):
    # Example: fetch data from Reddit API (placeholder)
    # Replace with actual logic
    response = requests.get(f"https://www.reddit.com/search.json?q={keyword}")
    return response.json()
