import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # MongoDB Configuration
    MONGODB_URI = os.getenv('MONGODB_URI')
    
    # Reddit API Configuration
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
    
    # Vector Search Configuration
    EMBEDDING_URL = os.getenv('EMBEDDING_URL')
    HF_TOKEN = os.getenv('HF_TOKEN')
    SUBREDDITS_FILE = 'data/subreddits.txt'