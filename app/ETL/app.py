import praw
import pymongo
from datetime import datetime
from typing import List
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

class RedditScraper:
    def __init__(self):
        # MongoDB setup
        try:
            self.client = pymongo.MongoClient("mongodb+srv://<username>:<password>@<your-cluster-url>")  # Replace with your connection string
            self.db = self.client['reddit_db']  # Creates 'reddit_db' database
            self.posts_collection = self.db['posts']  # Creates 'posts' collection
            
            # Test MongoDB connection
            self.client.server_info()
            print("Successfully connected to MongoDB!")
            
            # Create indexes
            self.posts_collection.create_index([('id', pymongo.ASCENDING)], unique=True)
            self.posts_collection.create_index([('subreddit', pymongo.ASCENDING)])
            self.posts_collection.create_index([('created_utc', pymongo.ASCENDING)])
            
        except Exception as e:
            print(f"MongoDB Connection Error: {e}")
            sys.exit(1)
            
        # Reddit API setup
        try:
            self.reddit = praw.Reddit(
                client_id="your_client_id",          # Replace with your client ID
                client_secret="your_client_secret",  # Replace with your client secret
                user_agent="your_user_agent"         # Replace with your user agent
            )
            print("Successfully connected to Reddit API!")
            
        except Exception as e:
            print(f"Reddit API Connection Error: {e}")
            sys.exit(1)

    def get_post_details(self, post) -> dict:
        """Extract detailed information from a post."""
        try:
            # Basic post information
            post_data = {
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
                
                # Additional post metadata
                'is_video': post.is_video if hasattr(post, 'is_video') else False,
                'is_original_content': post.is_original_content,
                'over_18': post.over_18,
                'spoiler': post.spoiler,
                'stickied': post.stickied,
                'locked': post.locked,
                'link_flair_text': post.link_flair_text,
                
                # Media information
                'media': post.media if hasattr(post, 'media') else None,
                'media_metadata': post.media_metadata if hasattr(post, 'media_metadata') else None,
                
                # Collection metadata
                'scraped_at': datetime.utcnow(),
                'last_updated': datetime.utcnow()
            }
            
            # Get comments
            comments = []
            post.comments.replace_more(limit=None)
            for comment in post.comments.list():
                try:
                    comment_data = {
                        'id': comment.id,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'body': comment.body,
                        'created_utc': datetime.fromtimestamp(comment.created_utc),
                        'score': comment.score,
                        'is_submitter': comment.is_submitter,
                        'parent_id': comment.parent_id,
                        'edited': comment.edited if hasattr(comment, 'edited') else False
                    }
                    comments.append(comment_data)
                except Exception as e:
                    print(f"Error processing comment {comment.id}: {e}")
                    continue
            
            post_data['comments'] = comments
            return post_data
            
        except Exception as e:
            print(f"Error getting post details: {e}")
            return None

    def scrape_subreddits(self):
        """Main function to scrape subreddits based on user input."""
        while True:
            # Get subreddit input
            print("\nEnter subreddit names (separated by commas) or 'quit' to exit:")
            user_input = input().strip().lower()
            
            if user_input == 'quit':
                break
                
            subreddits = [s.strip() for s in user_input.split(',')]
            
            # Get limit input
            print("Enter maximum number of posts per subreddit (press Enter for no limit):")
            limit_input = input().strip()
            limit = int(limit_input) if limit_input else None
            
            posts_processed = 0
            
            for subreddit_name in subreddits:
                print(f"\nProcessing r/{subreddit_name}...")
                
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Check if subreddit exists and is accessible
                    subreddit.id
                    
                    # Get posts from different sorts
                    for sort_type in ['hot', 'new', 'top']:
                        print(f"Getting {sort_type} posts...")
                        posts = getattr(subreddit, sort_type)(limit=limit)
                        
                        for post in posts:
                            try:
                                post_data = self.get_post_details(post)
                                
                                if post_data:
                                    # Update or insert post
                                    self.posts_collection.update_one(
                                        {'id': post_data['id']},
                                        {'$set': post_data},
                                        upsert=True
                                    )
                                    
                                    posts_processed += 1
                                    if posts_processed % 5 == 0:  # Progress update every 5 posts
                                        print(f"Processed {posts_processed} posts...")
                                        
                            except pymongo.errors.DuplicateKeyError:
                                continue  # Skip duplicates
                            except Exception as e:
                                print(f"Error processing post: {e}")
                                continue
                                
                except Exception as e:
                    print(f"Error accessing r/{subreddit_name}: {e}")
                    continue
            
            print(f"\nCompleted! Total posts processed: {posts_processed}")
            print(f"Posts are stored in the 'posts' collection of 'reddit_db' database")

if __name__ == "__main__":
    scraper = RedditScraper()
    scraper.scrape_subreddits()