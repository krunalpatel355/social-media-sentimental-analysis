#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd

# Define file paths
file1 = 'C:\\Users\\Hazel\\reddit_data.json'
file2 = 'C:\\Users\\Hazel\\redditdata1.json'

# Function to load JSON data into a DataFrame
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return pd.json_normalize(data)

# Load the datasets
df1 = load_json(file1)
df2 = load_json(file2)

# Combine the datasets into a single DataFrame
df = pd.concat([df1, df2], ignore_index=True)

# Display the first few rows to understand the structure
print("First few rows of the dataset:")
print(df.head())

# Show column names and types
print("\nColumns and data types:")
print(df.dtypes)

# Basic data cleaning: Drop rows or columns with all null values, if any
df = df.dropna(how='all', axis=1)  # Drop columns with all null values
df = df.dropna(how='all', axis=0)  # Drop rows with all null values

# Check for null values in remaining columns
print("\nNull values in each column:")
print(df.isnull().sum())

# Filling missing values (optional): Fill text columns with empty string, if needed
text_columns = ['title', 'selftext'] if 'title' in df.columns and 'selftext' in df.columns else []
for col in text_columns:
    df[col] = df[col].fillna('')

# Show a summary of basic statistics (for numerical columns)
print("\nSummary statistics for numerical columns:")
print(df.describe())

# Optional: Display the shape of the combined DataFrame
print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")


# In[6]:


import json
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load JSON data from both files
file1 = 'C:\\Users\\Hazel\\reddit_data.json'
file2 = 'C:\\Users\\Hazel\\redditdata1.json'

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return pd.json_normalize(data)

# Load and combine data
df1 = load_json(file1)
df2 = load_json(file2)
df = pd.concat([df1, df2], ignore_index=True)

# Choose text column
text_column = 'title' if 'title' in df.columns else 'selftext'
df[text_column] = df[text_column].fillna('')

# Define stop words and additional filters for politics and marketing terms
stop_words = set(stopwords.words('english'))
additional_stop_words = {'like', 'just', 'people', 'think', 'know', 'really', 'good', 'time', 'also', 'one', 'get', 'would', 'even', 'could', 'see', 'go', 'well', 'much', 'make', 'new', 'first', 'many', 'need'}
stop_words.update(additional_stop_words)

# Convert stop words to a list for CountVectorizer
stop_words = list(stop_words)

# Extract bigrams and trigrams using CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words=stop_words)
ngrams = vectorizer.fit_transform(df[text_column])

# Sum the frequencies of each bigram/trigram
ngram_counts = ngrams.sum(axis=0)
ngram_freq = [(word, ngram_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
ngram_freq = sorted(ngram_freq, key=lambda x: x[1], reverse=True)

# Get the top 10 bigrams/trigrams
top_ngrams = dict(ngram_freq[:10])

# Plot the top bigrams/trigrams
plt.figure(figsize=(10, 5))
plt.bar(top_ngrams.keys(), top_ngrams.values())
plt.title("Top 10 Trending Bigrams/Trigrams in Politics and Marketing")
plt.xlabel("Bigrams/Trigrams")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()


# In[7]:


import json
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize VADER SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Load JSON data from both files
file1 = 'C:\\Users\\Hazel\\reddit_data.json'
file2 = 'C:\\Users\\Hazel\\redditdata1.json'

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return pd.json_normalize(data)

# Load and combine data
df1 = load_json(file1)
df2 = load_json(file2)
df = pd.concat([df1, df2], ignore_index=True)

# Choose text column for sentiment analysis
text_column = 'title' if 'title' in df.columns else 'selftext'
df[text_column] = df[text_column].fillna('')

# Apply sentiment analysis
df['sentiment_score'] = df[text_column].apply(lambda text: sia.polarity_scores(text)['compound'])

# Categorize sentiment based on score
df['sentiment_category'] = pd.cut(df['sentiment_score'], bins=[-1, -0.05, 0.05, 1], labels=['Negative', 'Neutral', 'Positive'])

# Display results
print(df[['sentiment_score', 'sentiment_category', text_column]].head())

# Optional: Show sentiment distribution
df['sentiment_category'].value_counts().plot(kind='bar', title="Sentiment Distribution in Posts")
plt.xlabel("Sentiment Category")
plt.ylabel("Frequency")
plt.show()


# In[8]:


# Define topic keywords for segmentation
politics_keywords = ['election', 'vote', 'policy', 'government', 'president', 'campaign', 'law', 'rights']
marketing_keywords = ['brand', 'market', 'digital', 'strategy', 'advertising', 'social', 'content', 'seo', 'product']

# Function to categorize based on keywords
def categorize_by_topic(text):
    text = text.lower()
    if any(word in text for word in politics_keywords):
        return 'Politics'
    elif any(word in text for word in marketing_keywords):
        return 'Marketing'
    else:
        return 'Other'

# Apply topic-based segmentation
df['topic_segment'] = df[text_column].apply(categorize_by_topic)

# Define length-based segmentation
def categorize_by_length(text):
    length = len(text.split())
    if length < 20:
        return 'Short'
    elif 20 <= length <= 50:
        return 'Medium'
    else:
        return 'Long'

# Apply length-based segmentation
df['length_segment'] = df[text_column].apply(categorize_by_length)

# Display segmented data
print(df[['topic_segment', 'length_segment', text_column]].head())

# Visualization of topic segmentation
df['topic_segment'].value_counts().plot(kind='bar', title="Topic Segmentation")
plt.xlabel("Topic")
plt.ylabel("Frequency")
plt.show()

# Visualization of length segmentation
df['length_segment'].value_counts().plot(kind='bar', title="Length Segmentation")
plt.xlabel("Post Length Category")
plt.ylabel("Frequency")
plt.show()


# In[1]:


import json
import pandas as pd
import os

# Define file paths
file1 = 'C:\\Users\\Hazel\\reddit_data.json'
file2 = 'C:\\Users\\Hazel\\redditdata1.json'
file3 = 'C:\\Users\\Hazel\\redditdata2.json'

# Function to load JSON data into a DataFrame
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return pd.json_normalize(data)

# Load the datasets
df1 = load_json(file1)
df2 = load_json(file2)
df3 = load_json(file3)

# Display file sizes
print(f"File sizes:")
print(f"{file1}: {os.path.getsize(file1) / 1024 / 1024:.2f} MB")
print(f"{file2}: {os.path.getsize(file2) / 1024 / 1024:.2f} MB")
print(f"{file3}: {os.path.getsize(file3) / 1024 / 1024:.2f} MB")

# Combine the datasets into a single DataFrame
df = pd.concat([df1, df2, df3], ignore_index=True)

# Display the first few rows to understand the structure
print("\nFirst few rows of the combined dataset:")
print(df.head())

# Show column names and types
print("\nColumns and data types:")
print(df.dtypes)

# Basic data cleaning: Drop rows or columns with all null values, if any
df = df.dropna(how='all', axis=1)  # Drop columns with all null values
df = df.dropna(how='all', axis=0)  # Drop rows with all null values

# Check for null values in remaining columns
print("\nNull values in each column:")
print(df.isnull().sum())

# Filling missing values (optional): Fill text columns with empty string, if needed
text_columns = ['title', 'selftext'] if 'title' in df.columns and 'selftext' in df.columns else []
for col in text_columns:
    df[col] = df[col].fillna('')

# Show a summary of basic statistics (for numerical columns)
print("\nSummary statistics for numerical columns:")
print(df.describe())

# Optional: Display the shape of the combined DataFrame
print(f"\nCombined dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

