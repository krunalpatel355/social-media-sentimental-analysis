import praw
import pymongo
from datetime import datetime
from typing import Dict, List, Optional, NamedTuple
import sys
from tqdm.notebook import tqdm
from config.settings import Config
# MongoDB and Reddit Connection Setup

MONGODB_URI = Config.MONGODB_URI
REDDIT_CLIENT_ID = Config.REDDIT_CLIENT_ID
REDDIT_CLIENT_SECRET =  Config.REDDIT_CLIENT_SECRET
REDDIT_USER_AGENT = Config.REDDIT_USER_AGENT

class ConnectionManager:
    @staticmethod
    def setup_mongodb():
        """Set up MongoDB connection."""
        try:
            client = pymongo.MongoClient(MONGODB_URI)
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

# Comment Data Collector
class CommentCollector:
    @staticmethod
    def get_comment_data(comment) -> Dict:
        """Extract comment data."""
        return {
            'id': comment.id,
            'author': str(comment.author) if comment.author else '[deleted]',
            'body': comment.body,
            'created_utc': datetime.fromtimestamp(comment.created_utc),
            'score': comment.score,
            'is_submitter': comment.is_submitter,
            'parent_id': comment.parent_id,
            'edited': comment.edited if hasattr(comment, 'edited') else False
        }
    
    @staticmethod
    def get_comments(post, comment_limit: int) -> List[Dict]:
        """Get comments from a post."""
        comments = []
        try:
            post.comments.replace_more(limit=comment_limit)
            for comment in post.comments.list():
                try:
                    comment_data = CommentCollector.get_comment_data(comment)
                    comments.append(comment_data)
                except Exception as e:
                    print(f"Error processing comment {comment.id}: {e}")
                    continue
        except Exception as e:
            print(f"Error fetching comments: {e}")
        return comments

# Post Data Collector
class PostCollector:
    def __init__(self, include_comments: bool = True, comment_limit: int = 10):
        self.include_comments = include_comments
        self.comment_limit = comment_limit

    def get_post_data(self, post) -> Optional[Dict]:
        """Extract post data and optionally comments."""
        try:
            post_data = {
                # Basic information
                'id': post.id,
                'title': post.title,
                'author': str(post.author) if post.author else '[deleted]',
                'created_utc': datetime.fromtimestamp(post.created_utc),
                'score': post.score,
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments,
                'url': post.url,
                'selftext': post.selftext,
                'permalink': post.permalink,
                'subreddit': post.subreddit.display_name,
                
                # Metadata
                'is_video': post.is_video if hasattr(post, 'is_video') else False,
                'is_original_content': post.is_original_content,
                'over_18': post.over_18,
                'spoiler': post.spoiler,
                'stickied': post.stickied,
                'locked': post.locked,
                'link_flair_text': post.link_flair_text,
                
                # Media
                'media': post.media if hasattr(post, 'media') else None,
                'media_metadata': post.media_metadata if hasattr(post, 'media_metadata') else None,
                
                # Timestamps
                'scraped_at': datetime.utcnow(),
                'last_updated': datetime.utcnow()
            }
            
            # Add comments if enabled
            if self.include_comments:
                post_data['comments'] = CommentCollector.get_comments(post, self.comment_limit)
                
            return post_data
            
        except Exception as e:
            print(f"Error getting post details: {e}")
            return None

# MongoDB Index Manager
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


class ScraperConfig(NamedTuple):
    """Configuration class to store all user inputs"""
    subreddits: List[str]
    post_limit: Optional[int]
    include_comments: bool
    comment_limit: int
    sort_types: List[str] = ['hot', 'new', 'top']

def get_user_inputs() -> ScraperConfig:
    """Collect all user inputs at once"""
    print("\nWelcome to Reddit Scraper! Please provide the following information:")
    
    # Get subreddits
    print("\nEnter topic names (separated by comma):")
    subreddits = [s.strip() for s in input().strip().lower().split(',')]
    
    # Get post limit
    print("\nEnter maximum number of posts per subreddit (press Enter for no limit):")
    limit_input = input().strip()
    post_limit = int(limit_input) if limit_input else None
    
    # Get comment preferences
    print("\nDo you want to include comments? (yes/no):")
    include_comments = input().strip().lower() == 'yes'
    
    # Get comment limit if including comments
    comment_limit = 0
    if include_comments:
        print("\nEnter maximum number of comment trees to expand (default is 10):")
        comment_input = input().strip()
        comment_limit = int(comment_input) if comment_input else 10
    
    return ScraperConfig(
        subreddits=subreddits,
        post_limit=post_limit,
        include_comments=include_comments,
        comment_limit=comment_limit
    )

# [Previous ConnectionManager, CommentCollector, and PostCollector classes remain unchanged]

class RedditScraper:
    def __init__(self, config: ScraperConfig):
        """Initialize the Reddit Scraper with configuration."""
        self.config = config
        
        # Setup connections
        self.client, self.db, self.posts_collection = ConnectionManager.setup_mongodb()
        self.reddit = ConnectionManager.setup_reddit()
        
        # Create indexes
        index_manager = IndexManager(self.posts_collection)
        index_manager.create_indexes()
        
        # Initialize collectors
        self.post_collector = PostCollector(
            include_comments=config.include_comments,
            comment_limit=config.comment_limit
        )
    
    def scrape_subreddits(self):
        """Scrape posts from specified subreddits using the provided configuration."""
        posts_processed = 0
        
        for subreddit_name in self.config.subreddits:
            print(f"\nProcessing r/{subreddit_name}...")
            
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                subreddit.id  # Test if subreddit exists
                
                for sort_type in self.config.sort_types:
                    print(f"Getting {sort_type} posts...")
                    posts = getattr(subreddit, sort_type)(limit=self.config.post_limit)
                    
                    for post in tqdm(list(posts), desc=f"{sort_type} posts"):
                        try:
                            post_data = self.post_collector.get_post_data(post)
                            
                            if post_data:
                                self.posts_collection.update_one(
                                    {'id': post_data['id']},
                                    {'$set': post_data},
                                    upsert=True
                                )
                                posts_processed += 1
                                 
                        except pymongo.errors.DuplicateKeyError:
                            continue
                        except Exception as e:
                            print(f"Error processing post: {e}")
                            continue
                            
            except Exception as e:
                print(f"Error accessing r/{subreddit_name}: {e}")
                continue
        
        print(f"\n✅ Completed! Total posts processed: {posts_processed}")
        print(f"Posts are stored in the 'posts' collection of 'reddit_db' database")

def etl():
    """Main function to run the scraper with user configuration."""
    # Get all user inputs at once
    config = get_user_inputs()
    
    # Print configuration summary
    print("\nScraper Configuration:")
    print(f"Subreddits: {', '.join(config.subreddits)}")
    print(f"Posts per subreddit: {'Unlimited' if config.post_limit is None else config.post_limit}")
    print(f"Include comments: {config.include_comments}")
    if config.include_comments:
        print(f"Comment trees per post: {config.comment_limit}")
    print("\nStarting scraping process...")
    
    # Initialize and run scraper
    scraper = RedditScraper(config)
    scraper.scrape_subreddits()

if __name__ == "__main__":
    etl()