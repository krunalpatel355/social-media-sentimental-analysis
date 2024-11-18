# reddit_scraper.py
import praw
import pymongo
from datetime import datetime
from typing import List, Dict, Optional
import os

class RedditScraper:
    def __init__(self, include_comments: bool = False):
        self._setup_mongodb()
        self._setup_reddit()
        self.include_comments = include_comments

    def _setup_mongodb(self):
        """Set up MongoDB connection and indexes."""
        try:
            self.client = pymongo.MongoClient(os.getenv('MONGODB_URI'))
            self.db = self.client['reddit_db']
            self.posts_collection = self.db['posts']
            self.client.server_info()
            self._create_indexes()
        except Exception as e:
            raise Exception(f"MongoDB Connection Error: {e}")

    def _setup_reddit(self):
        """Set up Reddit API connection."""
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT')
            )
        except Exception as e:
            raise Exception(f"Reddit API Connection Error: {e}")

    def _create_indexes(self):
        """Create MongoDB indexes."""
        indexes = [
            ('id', pymongo.ASCENDING),
            ('subreddit', pymongo.ASCENDING),
            ('created_utc', pymongo.ASCENDING)
        ]
        for field, direction in indexes:
            self.posts_collection.create_index(
                [(field, direction)],
                unique=(field == 'id')
            )

    def get_subreddit_preview(self, subreddit_name: str) -> List[Dict]:
        """Get initial preview of subreddit posts."""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            for post in subreddit.hot(limit=10):
                posts.append({
                    'id': post.id,
                    'title': post.title,
                    'author': str(post.author) if post.author else '[deleted]',
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc).isoformat(),
                    'url': post.url,
                    'permalink': post.permalink
                })
            return posts
        except Exception as e:
            raise Exception(f"Error accessing subreddit: {e}")

    def get_more_posts(self, subreddit_name: str, last_post_id: str) -> List[Dict]:
        """Fetch more posts after the last post ID."""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            found_last = False
            
            for post in subreddit.hot(limit=50):
                if post.id == last_post_id:
                    found_last = True
                    continue
                if found_last and len(posts) < 10:
                    posts.append({
                        'id': post.id,
                        'title': post.title,
                        'author': str(post.author) if post.author else '[deleted]',
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc).isoformat(),
                        'url': post.url,
                        'permalink': post.permalink
                    })
            return posts
        except Exception as e:
            raise Exception(f"Error fetching more posts: {e}")

    def scrape_selected_posts(self, post_ids: List[str], include_comments: bool) -> List[Dict]:
        """Scrape full data for selected posts."""
        results = []
        for post_id in post_ids:
            try:
                post = self.reddit.submission(id=post_id)
                post_data = self._get_full_post_data(post, include_comments)
                if post_data:
                    self.posts_collection.update_one(
                        {'id': post_data['id']},
                        {'$set': post_data},
                        upsert=True
                    )
                    results.append({
                        'id': post_data['id'],
                        'title': post_data['title'],
                        'status': 'success'
                    })
            except Exception as e:
                results.append({
                    'id': post_id,
                    'status': 'error',
                    'message': str(e)
                })
        return results

    def _get_full_post_data(self, post, include_comments: bool) -> Dict:
        """Get complete post data including optional comments."""
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
            'is_video': post.is_video if hasattr(post, 'is_video') else False,
            'is_original_content': post.is_original_content,
            'over_18': post.over_18,
            'spoiler': post.spoiler,
            'stickied': post.stickied,
            'locked': post.locked,
            'link_flair_text': post.link_flair_text,
            'scraped_at': datetime.utcnow()
        }

        if include_comments:
            post_data['comments'] = self._get_comments(post)

        return post_data

    def _get_comments(self, post) -> List[Dict]:
        """Get formatted comment data from post."""
        comments = []
        try:
            post.comments.replace_more(limit=5)
            for comment in post.comments.list():
                comments.append({
                    'id': comment.id,
                    'author': str(comment.author) if comment.author else '[deleted]',
                    'body': comment.body,
                    'created_utc': datetime.fromtimestamp(comment.created_utc),
                    'score': comment.score,
                    'is_submitter': comment.is_submitter,
                    'parent_id': comment.parent_id
                })
        except Exception as e:
            print(f"Error fetching comments: {e}")
        return comments