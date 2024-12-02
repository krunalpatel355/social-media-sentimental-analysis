import praw
import pandas as pd
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os
from dotenv import load_dotenv
import faiss
import json

class RedditDataCollector:
    def __init__(self):
        load_dotenv()
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.posts_data = []

    def find_subreddits(self, keyword: str, limit: int = 10) -> List[str]:
        """Find relevant subreddits based on keyword"""
        # Search for subreddits directly
        subreddits = list(self.reddit.subreddits.search(keyword, limit=limit))
        
        # Extract unique subreddit names
        subreddit_names = []
        seen = set()
        
        for subreddit in subreddits:
            if subreddit.display_name not in seen:
                subreddit_names.append(subreddit.display_name)
                seen.add(subreddit.display_name)
        
        print(f"Found {len(subreddit_names)} subreddits for keyword: {keyword}")
        return subreddit_names

    def collect_posts(self, subreddits: List[str], posts_per_sub: int = 100) -> List[Dict]:
        """Collect posts from given subreddits"""
        total_posts = 0
        for subreddit_name in subreddits:
            try:
                print(f"Collecting posts from r/{subreddit_name}")
                subreddit = self.reddit.subreddit(subreddit_name)
                
                for post in subreddit.hot(limit=posts_per_sub):
                    post_data = {
                        'id': post.id,
                        'title': post.title,
                        'text': post.selftext,
                        'subreddit': subreddit_name,
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc).isoformat()
                    }
                    self.posts_data.append(post_data)
                    total_posts += 1
                
                print(f"Collected {total_posts} posts from r/{subreddit_name}")
            
            except Exception as e:
                print(f"Error collecting posts from r/{subreddit_name}: {str(e)}")
                continue
        
        print(f"Total posts collected: {len(self.posts_data)}")
        return self.posts_data

    def build_vector_index(self):
        """Build FAISS index from collected posts"""
        if not self.posts_data:
            raise ValueError("No posts collected yet. Call collect_posts() first.")

        print("Building vector index...")
        texts = [f"{post['title']} {post['text']}" for post in self.posts_data]
        embeddings = self.model.encode(texts)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Save index and data
        faiss.write_index(self.index, "posts_index.faiss")
        with open("posts_data.json", "w") as f:
            json.dump(self.posts_data, f)
        
        print(f"Index built and saved with {len(self.posts_data)} posts")

    def vector_search(self, query: str, k: int = 5, filter_by: str = None) -> List[Dict]:
        """Search for similar posts using vector similarity"""
        if not self.index:
            raise ValueError("Index not built yet. Call build_vector_index() first.")

        print(f"Searching for: {query}")
        query_vector = self.model.encode([query])
        
        D, I = self.index.search(query_vector.astype('float32'), k)
        
        results = []
        for idx in I[0]:
            post = self.posts_data[idx]
            if filter_by == 'most_comments':
                if post['num_comments'] > 10:  # Arbitrary threshold
                    results.append(post)
            else:
                results.append(post)
        
        results = sorted(results, 
                        key=lambda x: x['num_comments'] if filter_by == 'most_comments' else x['score'], 
                        reverse=True)
        
        print(f"Found {len(results)} matching posts")
        return results

# Example usage
def main():
    # Initialize collector
    collector = RedditDataCollector()
    
    # Find relevant subreddits
    keyword = "machine learning"
    print(f"\nSearching for subreddits related to '{keyword}'...")
    subreddits = collector.find_subreddits(keyword)
    
    # Collect posts
    print("\nCollecting posts from found subreddits...")
    posts = collector.collect_posts(subreddits)
    
    # Build search index
    print("\nBuilding search index...")
    collector.build_vector_index()
    
    # Search example
    query = "How to start with deep learning?"
    print(f"\nSearching for: '{query}'...")
    results = collector.vector_search(query, filter_by='most_comments')
    
    # Print some results
    print("\nTop results:")
    for i, post in enumerate(results[:3], 1):
        print(f"\n{i}. {post['title']}")
        print(f"Subreddit: r/{post['subreddit']}")
        print(f"Score: {post['score']}")
        print(f"Comments: {post['num_comments']}")

if __name__ == "__main__":
    main()