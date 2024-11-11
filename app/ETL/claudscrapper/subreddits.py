import pymongo
import requests
from typing import List
import json
import certifi
from VES.main import ETL

class RedditVectorSearch:
    def __init__(self, mongodb_uri: str, hf_token: str):
        try:
            # Initialize MongoDB connection with SSL certificate
            self.client = pymongo.MongoClient(
                mongodb_uri,
                tlsCAFile=certifi.where(),  # Add SSL certificate verification
                serverSelectionTimeoutMS=5000  # Reduce server selection timeout
            )
            # Test the connection
            self.client.server_info()
            self.db = self.client.redditdb
            self.collection = self.db.subreddits
            print("Successfully connected to MongoDB!")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            raise
        
        # HuggingFace settings
        self.hf_token = hf_token
        self.embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using HuggingFace API"""
        response = requests.post(
            self.embedding_url,
            headers={"Authorization": f"Bearer {self.hf_token}"},
            json={"inputs": text}
        )
        if response.status_code != 200:
            raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
        return response.json()

    def needs_initialization(self) -> bool:
        """Check if collection needs to be initialized"""
        try:
            return self.collection.count_documents({}) == 0
        except Exception as e:
            print(f"Error checking collection: {e}")
            raise

    def load_subreddits_from_file(self, file_path: str) -> None:
        """Load subreddit names from text file and create embeddings"""
        try:
            # Drop existing collection if exists
            self.collection.drop()
            
            # Create vector search index
            index_config = {
                "mappings": {
                    "dynamic": True,
                    "fields": {
                        "embedding": {
                            "dimensions": 384,
                            "similarity": "dotProduct",
                            "type": "knnVector"
                        }
                    }
                }
            }
            
            # Create the index
            self.collection.create_index([("embedding", "vectorSearch")], 
                                       vectorSearchOptions=index_config)
            
            # Read and process file
            with open(file_path, 'r') as file:
                for line in file:
                    # Split line into subreddit name and subscriber count
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        subreddit_name = parts[0]
                        subscribers = int(parts[1])
                        
                        # Generate embedding for subreddit name
                        embedding = self.generate_embedding(subreddit_name)
                        
                        # Insert document with embedding and subscriber count
                        doc = {
                            "subreddit": subreddit_name,
                            "subscribers": subscribers,
                            "embedding": embedding
                        }
                        self.collection.insert_one(doc)
            
            print("Subreddits loaded and indexed successfully!")
        except Exception as e:
            print(f"Error in load_subreddits_from_file: {e}")
            raise

    def search_similar_subreddits(self, query: str, limit: int = 10) -> str:
        """Search for similar subreddits based on query"""
        try:
            # Generate embedding for query
            query_embedding = self.generate_embedding(query)
            
            # Perform vector search
            results = self.collection.aggregate([
                {"$vectorSearch": {
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": limit * 2,
                    "limit": limit,
                    "index": "vectorSearch",
                }}
            ])
            
            # Extract subreddit names and subscriber counts from results
            similar_subreddits = [f"{doc['subreddit']} {doc['subscribers']}" for doc in results]
            
            # Return space-separated string of results
            return " ".join(similar_subreddits)
        except Exception as e:
            print(f"Error in search_similar_subreddits: {e}")
            raise

def main():
    try:
        # Configuration
        mongodb_uri = "mongodb+srv://beau:bngeFBqJJoEWqRNd@cluster0.svcxhgj.mongodb.net/?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true"
        hf_token = "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"
        subreddits_file = "subreddits.txt"
        
        # Initialize vector search
        vector_search = RedditVectorSearch(mongodb_uri, hf_token)
        
        # Check if initialization is needed
        if vector_search.needs_initialization():
            print("Initializing database with subreddit data...")
            vector_search.load_subreddits_from_file(subreddits_file)
        else:
            print("Database already initialized, proceeding with vector search...")
        
        # Interactive search loop
        while True:
            query = input("\nEnter your search query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            results = vector_search.search_similar_subreddits(query)
            print(f"\nSimilar subreddits: {results}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return

if __name__ == "__main__":
    main()