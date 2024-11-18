# data_collectors.py
from datetime import datetime
from typing import Dict, List, Optional

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