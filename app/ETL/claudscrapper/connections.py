# connections.py
import praw
import pymongo
import sys
from config import MONGODB_URI, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

class ConnectionManager:
    @staticmethod
    def setup_mongodb():
        """Set up MongoDB connection."""
        try:
            client = pymongo.MongoClient("mongodb+srv://krunalpatel35538:YHFyBoSvWR1hKXkB@cluster0.lu5p4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
            db = client['reddit_db']
            posts_collection = db['posts']
            
            # Test connection
            client.server_info()
            print("✅ Successfully connected to MongoDB!")
            
            return client, db, posts_collection
            
        except Exception as e:
            print(f"❌ MongoDB Connection Error: {e}")
            sys.exit(1)
    
    @staticmethod
    def setup_reddit():
        """Set up Reddit API connection."""
        try:
            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )
            print("✅ Successfully connected to Reddit API!")
            return reddit
            
        except Exception as e:
            print(f"❌ Reddit API Connection Error: {e}")
            sys.exit(1)

a= ConnectionManager()
a.setup_mongodb()