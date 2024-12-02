import requests
from typing import List
import numpy as np
from pymongo import MongoClient
from app.config.settings import Config
import pymongo
import certifi
from typing import Dict, Any

class DatabaseConnection:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None

    def connect(self):
        try:
            self.client = pymongo.MongoClient(
                Config.MONGODB_URI,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=5000
            )
            self.client.server_info()
            self.db = self.client['reddit_db']
            self.collection = self.db['subReddits']

            # print("Successfully connected to MongoDB!")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            raise

    def create_search_index(self):
        # Create a standard index on the embedding field
        self.collection.create_index([("embedding", pymongo.ASCENDING)], name="embedding_index")



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


class VES():
    def __init__(self,input_text):
        self.input_text = input_text
    
    def vector_search(self):
        try:
            vector_search = VectorSearch()

            if vector_search.needs_initialization():
                print("Initializing database with subreddit data...")
                vector_search.load_subreddits_from_file(Config.SUBREDDITS_FILE)
            # else:
            #     print("Database already initialized, proceeding with vector search...")
                

            # query = input("\nEnter your search query: ")
            results = vector_search.search_similar_subreddits(self.input_text)

            results = results.split()

            # def convert(results):
            #     results_dict = {}
            #     for i in range(0, len(results),2):
            #         results_dict[i] = results[i] +" "+ results[i+1]
            #     return results_dict

            # return convert(results)
            results = [results[i] for i in range(0,len(results),2)]
            return {
                # "options": [f"Option {i}" for i in range(1, 11)]
                "options":results
            }

        except Exception as e:
            print(f"An error occurred: {e}")
            return
        
search_query = "politics"
ves_instance = VES(search_query)
options = ves_instance.vector_search()
print(options)
    # input_text = "politics"

    # results = VES(input_text)

    # for i in results:
    #     print(results[i])