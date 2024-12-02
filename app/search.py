from typing import List, Optional, Dict, Union
from datetime import datetime, UTC
import uuid
from pymongo import MongoClient, TEXT
from dataclasses import dataclass, asdict
from bson import ObjectId
from app.config.settings import Config

# MongoDB Connection
MONGODB_URI = Config.MONGODB_URI
@dataclass
class SearchParameters:
    """Class to hold search parameters"""
    subreddits: List[str]
    from_time: Optional[datetime] = None
    to_time: Optional[datetime] = None
    sort_types: List[str] = None
    post_limit: Optional[int] = None
    include_comments: bool = False
    search_text: str = ""
    comment_limit: int = 10

class SearchManager:
    def __init__(self):
        """Initialize MongoDB connection and collections"""
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client['reddit_db']
        self.posts_collection = self.db['posts']
        self.search_collection = self.db['search']
        
        # Create indexes for both collections
        self._create_indexes()
    
    def _create_indexes(self):
        """Create necessary indexes for both collections"""
        try:
            # Indexes for search collection
            self.search_collection.create_index("search_id", unique=True)
            self.search_collection.create_index("timestamp")
            
            # Text index for posts collection
            self.posts_collection.create_index([
                ("title", TEXT),
                ("selftext", TEXT),
                ("subreddit", TEXT)
            ])
            print("✅ Successfully created all indexes!")
        except Exception as e:
            print(f"Error creating indexes: {e}")

    def _build_query(self, parameters: SearchParameters) -> Dict:
        """Build MongoDB query based on search parameters"""
        query = {}
        
        # Add subreddit filter
        if parameters.subreddits:
            query['subreddit'] = {'$in': parameters.subreddits}
        
        # Add time range filter if provided
        if parameters.from_time or parameters.to_time:
            time_query = {}
            if parameters.from_time:
                time_query['$gte'] = parameters.from_time
            if parameters.to_time:
                time_query['$lte'] = parameters.to_time
            if time_query:
                query['created_utc'] = time_query
        
        # Add text search if provided
        if parameters.search_text:
            query['$text'] = {'$search': parameters.search_text}
        
        return query

    def simple_search(self, subreddits: Union[str, List[str]]) -> str:
        """
        Perform simple search based on subreddits only
        Returns: search_id for future reference
        """
        # Convert single subreddit to list if necessary
        if isinstance(subreddits, str):
            subreddits = [subreddits]
        
        # Create search parameters with only subreddits
        parameters = SearchParameters(
            subreddits=subreddits
        )
        
        # Perform search
        query = {'subreddit': {'$in': subreddits}}
        post_ids = [str(post['_id']) for post in self.posts_collection.find(query, {'_id': 1})]
        
        # Create search record
        search_record = {
            'search_id': str(uuid.uuid4()),
            'timestamp': datetime.now(UTC),  # Using timezone-aware datetime
            'parameters': None,  # Simple search has no parameters
            'post_ids': post_ids,
            'total_posts': len(post_ids),
            'search_type': 'simple'
        }
        
        # Store search record
        self.search_collection.insert_one(search_record)
        
        return search_record['search_id']

    def advanced_search(self, parameters: SearchParameters) -> str:
        """
        Perform advanced search based on provided parameters
        Returns: search_id for future reference
        """
        # Build query from parameters
        query = self._build_query(parameters)
        
        # Perform search
        post_ids = [str(post['_id']) for post in self.posts_collection.find(query, {'_id': 1})]
        
        # Create search record
        search_record = {
            'search_id': str(uuid.uuid4()),
            'timestamp': datetime.now(UTC),  # Using timezone-aware datetime
            'parameters': asdict(parameters),
            'post_ids': post_ids,
            'total_posts': len(post_ids),
            'search_type': 'advanced'
        }
        
        # Store search record
        self.search_collection.insert_one(search_record)
        
        return search_record['search_id']

    def get_search_results(self, search_id: str) -> Dict:
        """Retrieve search results by search_id"""
        result = self.search_collection.find_one({'search_id': search_id})
        if result:
            # Convert ObjectId to string for JSON serialization
            result['_id'] = str(result['_id'])
            return result
        return None

class search:
    def __init__(self, parameters):
        self.parameters = parameters

    def perform_simple_search(self):
        search_manager = SearchManager()
        
        # Example 1: Simple Search
        print("\nPerforming simple search...")
        simple_search_id = search_manager.simple_search(self.parameters)
        simple_results = search_manager.get_search_results(simple_search_id)
        return simple_search_id
    
    def perform_advance_search(self):
        search_manager = SearchManager()
        # Example 2: Advanced Search
        print("\nPerforming advanced search...")
        
    
        advanced_search_id = search_manager.advanced_search(self.parameters)
        advanced_results = search_manager.get_search_results(advanced_search_id)
        return advanced_search_id



if __name__ == "__main__":
    parameters = SearchParameters(
            subreddits=['politics'],
            from_time=datetime(2024, 3, 1, tzinfo=UTC),  # Added timezone info
            to_time=datetime.now(UTC),
            sort_types=['hot', 'new', 'top'],
            post_limit=100,
            include_comments=False,
            search_text="election",
            comment_limit=10
        )
    s = search(parameters)
    # s.perform_simple_search()
    s.perform_advance_search()