import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import logging
import warnings
from datetime import datetime, timezone
warnings.filterwarnings('ignore')
from app.config.settings import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RedditAnalyzer:
    def __init__(self, mongo_uri: str):
        """
        Initialize MongoDB connection and load necessary data
        
        Args:
            mongo_uri (str): MongoDB connection string
        """
        try:
            # MongoDB Connection
            self.client = MongoClient(mongo_uri)
            self.db = self.client['reddit_db']
            self.posts_collection = self.db['posts']
            self.search_collection = self.db['search']
            
            # Test connection
            self.client.server_info()
            logger.info("Successfully connected to MongoDB")
            
            # Initialize NLTK resources
            self._initialize_nltk()
            
        except Exception as e:
            logger.error(f"Failed to initialize RedditAnalyzer: {str(e)}")
            raise
    
    def _initialize_nltk(self):
        """Initialize NLTK resources"""
        try:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))
    
    def _get_posts_by_search_id(self, search_id: str) -> List[Dict]:
        """
        Retrieve posts based on search_id
        
        Args:
            search_id (str): The search ID to retrieve posts for
            
        Returns:
            List[Dict]: List of posts
        """
        try:
            # Get search result
            search_result = self.search_collection.find_one({'search_id': search_id})
            if not search_result:
                logger.error(f"No search results found for ID: {search_id}")
                return []
            
            # Convert string IDs to ObjectID
            try:
                post_ids = [ObjectId(pid) for pid in search_result['post_ids']]
            except Exception as e:
                logger.error(f"Error converting post IDs: {str(e)}")
                return []
            
            # Fetch posts
            posts = list(self.posts_collection.find({'_id': {'$in': post_ids}}))
            logger.info(f"Retrieved {len(posts)} posts for search ID {search_id}")
            
            return posts
            
        except Exception as e:
            logger.error(f"Error retrieving posts: {str(e)}")
            return []

    def perform_sentiment_analysis(self, search_id: str) -> Dict:
        """
        Perform sentiment analysis on posts
        
        Args:
            search_id (str): The search ID to analyze
            
        Returns:
            Dict: Sentiment analysis results
        """
        posts = self._get_posts_by_search_id(search_id)
        if not posts:
            return {"error": "No posts found for analysis"}

        sentiment_results = {
            'overall': {'positive': 0, 'negative': 0, 'neutral': 0},
            'posts': [],
            'time_series': {},
            'subreddit_sentiment': {}
        }

        try:
            for post in posts:
                # Analyze post title and content
                text = f"{post['title']} {post.get('selftext', '')}"
                sentiment = TextBlob(text)
                
                # Determine sentiment category
                polarity = sentiment.sentiment.polarity
                if polarity > 0.1:
                    category = 'positive'
                elif polarity < -0.1:
                    category = 'negative'
                else:
                    category = 'neutral'
                
                # Convert timestamp to datetime if it's not already
                created_utc = post['created_utc']
                if isinstance(created_utc, (int, float)):
                    created_utc = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                
                # Store individual post sentiment
                post_sentiment = {
                    'id': str(post['_id']),
                    'title': post['title'],
                    'polarity': polarity,
                    'category': category,
                    'created_utc': created_utc.isoformat()
                }
                sentiment_results['posts'].append(post_sentiment)
                
                # Update overall counts
                sentiment_results['overall'][category] += 1
                
                # Update time series data
                date_key = created_utc.strftime('%Y-%m-%d')
                if date_key not in sentiment_results['time_series']:
                    sentiment_results['time_series'][date_key] = {
                        'positive': 0, 'neutral': 0, 'negative': 0
                    }
                sentiment_results['time_series'][date_key][category] += 1
                
                # Update subreddit sentiment
                subreddit = post['subreddit']
                if subreddit not in sentiment_results['subreddit_sentiment']:
                    sentiment_results['subreddit_sentiment'][subreddit] = {
                        'positive': 0, 'neutral': 0, 'negative': 0
                    }
                sentiment_results['subreddit_sentiment'][subreddit][category] += 1
            
            # Calculate percentages
            total_posts = len(posts)
            sentiment_results['overall'].update({
                f'{category}_percentage': (count / total_posts * 100)
                for category, count in sentiment_results['overall'].items()
            })
            
            return sentiment_results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {"error": f"Error performing sentiment analysis: {str(e)}"}

    def perform_segmentation_analysis(self, search_id: str, n_clusters: int = 5) -> Dict:
        """
        Perform text segmentation analysis using K-means clustering
        
        Args:
            search_id (str): The search ID to analyze
            n_clusters (int): Number of clusters to create
            
        Returns:
            Dict: Segmentation analysis results
        """
        posts = self._get_posts_by_search_id(search_id)
        if not posts:
            return {"error": "No posts found for analysis"}

        try:
            # Prepare text data
            texts = [f"{post['title']} {post.get('selftext', '')}" for post in posts]
            
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(posts)), random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Analyze clusters
            cluster_results = {
                'clusters': [],
                'cluster_stats': {},
                'feature_importance': {}
            }
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Analyze each cluster
            for i in range(kmeans.n_clusters):
                cluster_posts = [posts[j] for j in range(len(posts)) if clusters[j] == i]
                
                # Get top terms for cluster
                cluster_center = kmeans.cluster_centers_[i]
                top_terms_idx = cluster_center.argsort()[-10:][::-1]
                top_terms = [feature_names[idx] for idx in top_terms_idx]
                
                cluster_info = {
                    'cluster_id': i,
                    'size': len(cluster_posts),
                    'top_terms': top_terms,
                    'posts': [{
                        'id': str(post['_id']),
                        'title': post['title'],
                        'subreddit': post['subreddit'],
                        'score': post.get('score', 0)
                    } for post in cluster_posts[:5]]
                }
                
                cluster_results['clusters'].append(cluster_info)
                
                # Calculate cluster statistics
                cluster_results['cluster_stats'][f'cluster_{i}'] = {
                    'size': len(cluster_posts),
                    'size_percentage': len(cluster_posts) / len(posts) * 100,
                    'avg_score': np.mean([post.get('score', 0) for post in cluster_posts])
                }
            
            return cluster_results
            
        except Exception as e:
            logger.error(f"Error in segmentation analysis: {str(e)}")
            return {"error": f"Error performing segmentation analysis: {str(e)}"}

    def text_search(self, search_id: str, query: str) -> Dict:
        """
        Perform text search within the posts
        
        Args:
            search_id (str): The search ID to search within
            query (str): Search query
            
        Returns:
            Dict: Search results
        """
        posts = self._get_posts_by_search_id(search_id)
        if not posts:
            return {"error": "No posts found for search"}

        try:
            # Prepare search results
            search_results = {
                'query': query,
                'total_posts': len(posts),
                'matching_posts': []
            }

            # Tokenize query
            query_tokens = set(word_tokenize(query.lower())) - self.stop_words

            for post in posts:
                # Combine title and text
                post_text = f"{post['title']} {post.get('selftext', '')}".lower()
                post_tokens = set(word_tokenize(post_text)) - self.stop_words
                
                # Calculate relevance score (token overlap)
                overlap = len(query_tokens & post_tokens)
                if overlap > 0:
                    relevance_score = overlap / len(query_tokens)
                    
                    # Convert timestamp to datetime if it's not already
                    created_utc = post['created_utc']
                    if isinstance(created_utc, (int, float)):
                        created_utc = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                    
                    search_results['matching_posts'].append({
                        'id': str(post['_id']),
                        'title': post['title'],
                        'subreddit': post['subreddit'],
                        'score': post.get('score', 0),
                        'relevance_score': relevance_score,
                        'created_utc': created_utc.isoformat()
                    })

            # Sort by relevance score
            search_results['matching_posts'].sort(
                key=lambda x: x['relevance_score'],
                reverse=True
            )
            
            search_results['total_matches'] = len(search_results['matching_posts'])
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
            return {"error": f"Error performing text search: {str(e)}"}

class DASHBOARD:
    def __init__(self, search_id: str):
        """
        Initialize Dashboard with a specific search ID
        
        Args:
            search_id (str): Unique identifier for the search
        """
        self.search_id = search_id
        # self.logger = logging.getLogger(__name__)
        
    def get_sentiment_analyzer(self, mongo_uri: str) -> 'RedditAnalyzer':
        """
        Initialize and return RedditAnalyzer
        
        Args:
            mongo_uri (str): MongoDB connection string
        
        Returns:
            RedditAnalyzer: Initialized analyzer instance
        """
        try:
            analyzer = RedditAnalyzer(mongo_uri)
            # self.logger.info(f"RedditAnalyzer initialized for search_id: {self.search_id}")
            return analyzer
        except Exception as e:
            # self.logger.error(f"Failed to initialize RedditAnalyzer: {str(e)}")
            raise

    def perform_comprehensive_analysis(self) -> Dict[str, any]:
        """
        Perform comprehensive analysis including sentiment and segmentation
        
        Returns:
            Dict: Comprehensive analysis results
        """
        try:
            # Initialize analyzer
            analyzer = self.get_sentiment_analyzer(Config.MONGODB_URI)
            
            # Perform sentiment analysis
            self.logger.info("Starting sentiment analysis...")
            sentiment_results = analyzer.perform_sentiment_analysis(self.search_id)
            
            # Perform segmentation analysis
            self.logger.info("Starting segmentation analysis...")
            segmentation_results = analyzer.perform_segmentation_analysis(self.search_id)
            
            # Optional: Text search
            self.logger.info("Performing text search...")
            text_search_results = analyzer.text_search(self.search_id, "default query")
            
            return {
                "sentiment": sentiment_results,
                "segmentation": segmentation_results,
                "text_search": text_search_results
            }
        
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {str(e)}")
            return {
                "error": str(e),
                "message": "Failed to perform comprehensive analysis"
            }

