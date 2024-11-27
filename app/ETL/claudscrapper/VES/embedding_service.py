# embedding_service.py
import requests
from typing import List
from VES.config import Config

class EmbeddingService:
    @staticmethod
    def generate_embedding(text: str) -> List[float]:
        response = requests.post(
            Config.EMBEDDING_URL,
            headers={"Authorization": f"Bearer {Config.HF_TOKEN}"},
            json={"inputs": text}
        )
        if response.status_code != 200:
            raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
        return response.json()