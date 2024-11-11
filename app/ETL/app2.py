import praw
import pymongo
from datetime import datetime
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import sys
from tqdm.notebook import tqdm  # For better progress tracking

# Load environment variables
load_dotenv()

class RedditScraper:
    def __init__(self, include_comments: bool = True, comment_limit: int = 10):
        """
        Initialize the Reddit Scraper with configurable options.
        
        Args:
            include_comments: Whether to scrape comments or not
            comment_limit: Maximum number of comment trees to expand
        """
        self.include_comments = include_comments
        self.comment_limit = comment_limit
        self._setup_mongodb()
        self._setup_reddit()
        
    def _setup_mongodb(self):
        """Set up MongoDB connection and indexes."""
        try:
            self.client = pymongo.MongoClient(os.getenv('MONGODB_URI'))
            self.db = self.client['reddit_db']
            self.posts_collection = self.db['posts']
            
            # Test connection
            self.client.server_info()
            print("✅ Successfully connected to MongoDB!")
            
            # Create indexes
            self._create_indexes()
            
        except Exception as e:
            print(f"❌ MongoDB Connection Error: {e}")
            sys.exit(1)
    
    def _setup_reddit(self):
        """Set up Reddit API connection."""
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT')
            )
            print("✅ Successfully connected to Reddit API!")
            
        except Exception as e:
            print(f"❌ Reddit API Connection Error: {e}")
            sys.exit(1)
    
    def _create_indexes(self):
        """Create MongoDB indexes for better query performance."""
        indexes = [
            ('id', pymongo.ASCENDING),
            ('subreddit', pymongo.ASCENDING),
            ('created_utc', pymongo.ASCENDING)
        ]
        for field, direction in indexes:
            try:
                self.posts_collection.create_index(
                    [(field, direction)],
                    unique=(field == 'id')
                )
            except Exception as e:
                print(f"Warning: Failed to create index for {field}: {e}")

    def _get_comment_data(self, comment) -> Dict:
        """
        Extract comment data.
        
        Args:
            comment: PRAW comment object
            
        Returns:
            dict: Formatted comment data
        """
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

    def _get_post_data(self, post) -> Optional[Dict]:
        """
        Extract post data and optionally comments.
        
        Args:
            post: PRAW post object
            
        Returns:
            dict: Formatted post data with optional comments
        """
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
                post_data['comments'] = self._get_comments(post)
                
            return post_data
            
        except Exception as e:
            print(f"Error getting post details: {e}")
            return None

    def _get_comments(self, post) -> List[Dict]:
        """
        Get comments from a post.
        
        Args:
            post: PRAW post object
            
        Returns:
            list: List of comment dictionaries
        """
        comments = []
        try:
            post.comments.replace_more(limit=self.comment_limit)
            for comment in post.comments.list():
                try:
                    comment_data = self._get_comment_data(comment)
                    comments.append(comment_data)
                except Exception as e:
                    print(f"Error processing comment {comment.id}: {e}")
                    continue
        except Exception as e:
            print(f"Error fetching comments: {e}")
        return comments

    def scrape_subreddits(self, subreddits: Optional[List[str]] = None, 
                         post_limit: Optional[int] = None,
                         sort_types: List[str] = ['hot', 'new', 'top']):
        """
        Scrape posts from specified subreddits.
        
        Args:
            subreddits: List of subreddit names. If None, will prompt for input
            post_limit: Maximum posts per subreddit per sort type
            sort_types: Types of post sorts to collect
        """
        if subreddits is None:
            print("\nEnter subreddit names (separated by commas):")
            user_input = input().strip().lower()
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
                    
                    # Use tqdm for progress tracking
                    for post in tqdm(list(posts), desc=f"{sort_type} posts"):
                        try:
                            post_data = self._get_post_data(post)
                            
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

def main():
    """Main function to run the scraper with user configuration."""
    print("Do you want to include comments? (yes/no):")
    include_comments = input().strip().lower() == 'yes'
    
    if include_comments:
        print("Enter maximum number of comment trees to expand (default is 10):")
        comment_input = input().strip()
        comment_limit = int(comment_input) if comment_input else 10
    else:
        comment_limit = 0
    
    scraper = RedditScraper(include_comments=include_comments, 
                           comment_limit=comment_limit)
    scraper.scrape_subreddits()

if __name__ == "__main__":
    main()