# vector_search.py
from typing import List
import numpy as np
from VES.database import DatabaseConnection
from VES.embedding_service import EmbeddingService

class VectorSearch:
    def __init__(self):
        self.db = DatabaseConnection()
        self.db.connect()
        self.embedding_service = EmbeddingService()

    def needs_initialization(self) -> bool:
        try:
            return self.db.collection.count_documents({}) == 0
        except Exception as e:
            print(f"Error checking collection: {e}")
            raise

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def load_subreddits_from_file(self, file_path: str) -> None:
        try:
            self.db.collection.drop()
            self.db.create_search_index()
            
            batch_size = 100
            documents = []
            
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        subreddit_name = parts[0]
                        subscribers = int(parts[1])
                        
                        embedding = self.embedding_service.generate_embedding(subreddit_name)
                        
                        doc = {
                            "subreddit": subreddit_name,
                            "subscribers": subscribers,
                            "embedding": embedding
                        }
                        documents.append(doc)
                        
                        # Insert in batches for better performance
                        if len(documents) >= batch_size:
                            self.db.collection.insert_many(documents)
                            documents = []
                
                # Insert any remaining documents
                if documents:
                    self.db.collection.insert_many(documents)
            
            print("Subreddits loaded and indexed successfully!")
        except Exception as e:
            print(f"Error in load_subreddits_from_file: {e}")
            raise

    def search_similar_subreddits(self, query: str, limit: int = 10) -> str:
        try:
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Fetch all documents and calculate similarity in Python
            all_docs = list(self.db.collection.find({}, {"subreddit": 1, "subscribers": 1, "embedding": 1}))
            
            # Calculate similarities
            similarities = []
            for doc in all_docs:
                similarity = self.cosine_similarity(query_embedding, doc["embedding"])
                similarities.append((similarity, doc))
            
            # Sort by similarity and get top results
            similarities.sort(reverse=True)
            top_results = similarities[:limit]
            
            # Format results
            similar_subreddits = [
                f"{doc['subreddit']} {doc['subscribers']}" 
                for _, doc in top_results
            ]
            
            return " ".join(similar_subreddits)
        except Exception as e:
            print(f"Error in search_similar_subreddits: {e}")
            raise

# Add to requirements.txt:
# numpy>=1.19.2
