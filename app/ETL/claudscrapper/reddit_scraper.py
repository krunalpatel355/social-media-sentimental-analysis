# reddit_scraper.py
from typing import List, Optional
from tqdm.notebook import tqdm
import pymongo
from connections import ConnectionManager
from indexing import IndexManager
from data_collectors import PostCollector
from VES.main import ETL
    
class RedditScraper:
    def __init__(self, include_comments: bool = True, comment_limit: int = 10):
        """Initialize the Reddit Scraper with configurable options."""
        self.include_comments = include_comments
        self.comment_limit = comment_limit
        
        # Setup connections
        self.client, self.db, self.posts_collection = ConnectionManager.setup_mongodb()
        self.reddit = ConnectionManager.setup_reddit()
        
        # Create indexes
        index_manager = IndexManager(self.posts_collection)
        index_manager.create_indexes()
        
        # Initialize collectors
        self.post_collector = PostCollector(include_comments, comment_limit)
    
    def scrape_subreddits(self, subreddits: Optional[List[str]] = None, 
                         post_limit: Optional[int] = None,
                         sort_types: List[str] = ['hot', 'new', 'top']):
        """Scrape posts from specified subreddits."""
    

        if subreddits is None:
            print("\nEnter topic name:")
            user_input = input().strip().lower()
            results = ETL()
            print(results)
            subreddits = [s.strip() for s in user_input.split(',')]
        
        if post_limit is None:
            print("Enter maximum number of posts per subreddit (press Enter for no limit):")
            limit_input = input().strip()
            post_limit = int(limit_input) if limit_input else None
        
        posts_processed = 0
        
        for subreddit_name in subreddits:
            print(f"\nProcessing r/{subreddit_name}...")
            
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                subreddit.id  # Test if subreddit exists
                
                for sort_type in sort_types:
                    print(f"Getting {sort_type} posts...")
                    posts = getattr(subreddit, sort_type)(limit=post_limit)
                    
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
