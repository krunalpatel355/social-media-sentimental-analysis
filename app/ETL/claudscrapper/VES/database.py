# database.py
import pymongo
import certifi
from typing import Dict, Any
from VES.config import Config

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

            print("Successfully connected to MongoDB!")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            raise

    def create_search_index(self):
        # Create a standard index on the embedding field
        self.collection.create_index([("embedding", pymongo.ASCENDING)], name="embedding_index")
