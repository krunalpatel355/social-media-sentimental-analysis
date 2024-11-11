# app.py
from flask import Flask, render_template, request, jsonify
from reddit_scraper import RedditScraper
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
scraper = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_subreddit', methods=['POST'])
def search_subreddit():
    subreddit = request.json.get('query', '').strip()
    try:
        # Initialize scraper if not already done
        global scraper
        if not scraper:
            scraper = RedditScraper(include_comments=False)
        
        # Get preliminary subreddit data
        posts = scraper.get_subreddit_preview(subreddit)
        return jsonify({
            'status': 'success',
            'posts': posts
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/fetch_more', methods=['POST'])
def fetch_more():
    subreddit = request.json.get('subreddit')
    last_post_id = request.json.get('lastPostId')
    try:
        posts = scraper.get_more_posts(subreddit, last_post_id)
        return jsonify({
            'status': 'success',
            'posts': posts
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/scrape_selected', methods=['POST'])
def scrape_selected():
    data = request.json
    post_ids = data.get('selectedPosts', [])
    include_comments = data.get('includeComments', False)
    try:
        results = scraper.scrape_selected_posts(post_ids, include_comments)
        return jsonify({
            'status': 'success',
            'results': results
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)