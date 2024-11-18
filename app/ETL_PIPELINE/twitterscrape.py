#!/usr/bin/env python
# coding: utf-8

# In[2]:


import snscrape.modules.twitter as sntwitter
import pandas as pd
from textblob import TextBlob  # For sentiment analysis


# In[3]:


import snscrape.modules.twitter as sntwitter
import pandas as pd
from textblob import TextBlob
import time
import os

# Define comprehensive list of topics, including marketing trends
queries = [
    "politics",
    "election",
    "government policy",
    "marketing trends",
    "digital marketing",
    "consumer behavior",
    "social media trends",
    "content marketing",
    "influencer marketing",
    "video marketing",
    "SEO trends",
    "e-commerce trends",
    "branding strategies",
    "email marketing trends",
    "customer experience",
    "personalized marketing",
    "marketing automation",
    "mobile marketing",
    "data-driven marketing",
    "customer loyalty",
    "brand storytelling",
    "user-generated content",
    "viral marketing",
    "lead generation",
    "AI in marketing",
    "sustainable marketing",
    "omnichannel marketing",
    "emerging technology",
    "artificial intelligence",
    "climate change",
    "cryptocurrency",
    "metaverse",
    "renewable energy",
    "global economy",
    "healthcare innovation"
]
max_tweets_per_query = 500  # Number of tweets per query in each run
target_file_size_mb = 500  # Target file size in MB
save_path = 'all_topics_tweets_sentiment.csv'

# Load previously scraped data if it exists
if os.path.exists(save_path):
    all_tweets = pd.read_csv(save_path)
else:
    all_tweets = pd.DataFrame(columns=['target', 'ids', 'date', 'flag', 'user', 'text'])

# Function to check file size
def file_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)

# Track scraped tweet IDs to avoid duplicates
scraped_ids = set(all_tweets['ids']) if not all_tweets.empty else set()

# Loop over each query and scrape tweets
for query in queries:
    print(f"Scraping tweets for topic: {query}")

    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{query} lang:en').get_items()):
        # Stop if we've reached the maximum tweets or file size
        if i >= max_tweets_per_query or file_size_mb(save_path) >= target_file_size_mb:
            break
        
        # Skip duplicate tweets
        if tweet.id in scraped_ids:
            continue

        # Determine sentiment polarity and assign target
        polarity = TextBlob(tweet.content).sentiment.polarity
        if polarity > 0:
            target = 4  # Positive
        elif polarity < 0:
            target = 0  # Negative
        else:
            target = 2  # Neutral

        # Append tweet details
        tweets.append([target, tweet.id, tweet.date, query, tweet.user.username, tweet.content])
        
        # Add tweet ID to set of scraped IDs
        scraped_ids.add(tweet.id)

        # Progress update every 100 tweets
        if i % 100 == 0:
            current_size = file_size_mb(save_path)
            print(f"Progress Update: {len(scraped_ids)} tweets saved. Current file size: {current_size:.2f} MB")
            # Save periodically to avoid data loss
            all_tweets.to_csv(save_path, index=False)

    # Convert list to DataFrame
    df = pd.DataFrame(tweets, columns=['target', 'ids', 'date', 'flag', 'user', 'text'])

    # Append to main DataFrame and drop duplicates
    all_tweets = pd.concat([all_tweets, df]).drop_duplicates(subset=['ids'], keep='first').reset_index(drop=True)

    # Save progress after each query
    all_tweets.to_csv(save_path, index=False)
    print(f"Saved progress for topic '{query}' - Total tweets: {len(all_tweets)} - Current file size: {file_size_mb(save_path):.2f} MB")

    # Optional: Pause to avoid rate limits or being flagged
    time.sleep(5)

    # Stop if target file size is reached
    if file_size_mb(save_path) >= target_file_size_mb:
        print("Target file size reached, stopping scraping.")
        break

print("Scraping completed or target file size reached.")


# In[4]:


get_ipython().run_line_magic('pip', 'install --upgrade certifi')


# In[6]:


import snscrape.base
snscrape.base.Scraper._request_kwargs = {'verify': False}


# In[ ]:




