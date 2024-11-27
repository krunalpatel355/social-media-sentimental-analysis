# indexing.py
import pymongo
from typing import List, Tuple

class IndexManager:
    def __init__(self, collection):
        self.collection = collection
    
    def create_indexes(self):
        """Create MongoDB indexes for better query performance."""
        indexes: List[Tuple[str, int]] = [
            ('id', pymongo.ASCENDING),
            ('subreddit', pymongo.ASCENDING),
            ('created_utc', pymongo.ASCENDING)
        ]
        for field, direction in indexes:
            try:
                self.collection.create_index(
                    [(field, direction)],
                    unique=(field == 'id')
                )
            except Exception as e:
                print(f"Warning: Failed to create index for {field}: {e}")