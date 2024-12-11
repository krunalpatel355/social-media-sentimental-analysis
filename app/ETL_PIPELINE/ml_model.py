# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestClassifier
import spacy

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

# Step 2: Define Dataset
col_name = ['id', 'brand', 'emotion', 'tweet']
df = pd.read_csv('twitter_training.csv', names=col_name)

# Step 2.1: Check dataset information
df.info()

# Step 3: Preprocessing
# 3.1 Check for nulls
df[df.tweet.isna() == True]

# 3.2 Drop null values
df.dropna(inplace=True)

# 3.3 Drop duplicates
df.drop_duplicates(inplace=True)

# 3.4 Define preprocessing function to remove stop words and lemmatize
def preprocess(text):
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(filtered_tokens)

# 3.5 Apply function to data
df['textPreprocessed'] = df['tweet'].apply(preprocess)

# Step 4: EDA
# 4.1 Univariate Analysis
# Brand Distribution
value_counts = df['brand'].value_counts()
plt.figure(figsize=(12, 12))
plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Category Distribution")
plt.show()

# Emotion Analysis
value_counts = df['emotion'].value_counts()
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.bar(value_counts.index, value_counts, color='skyblue')
plt.title('Emotion Counts')
plt.xlabel('Emotion')
plt.ylabel('Count')

plt.subplot(2, 1, 2)
plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Emotion Distribution')
plt.tight_layout()
plt.show()

# Word Clouds for Different Emotions
for emotion in df['emotion'].unique():
    reviews = df[df['emotion'] == emotion]['textPreprocessed']
    text = ' '.join(reviews)
    wordcloud = WordCloud(width=1500, height=800, background_color='black', min_font_size=15).generate(text)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud)
    plt.title(f'Word Cloud for {emotion} Reviews')
    plt.axis('off')
    plt.show()

# 4.2 Bivariate Analysis
reactions_entities = pd.crosstab(df['brand'], df['emotion'])
reactions_entities.plot(kind='bar', figsize=(16, 6), grid=True)
plt.show()

# Step 5: Train-Test Split
# 5.1 Drop unnecessary columns
x = df['textPreprocessed']
y = df['emotion']

# Encode categorical data
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 5.2 Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Create and Train the Pipeline
randomforest = RandomForestClassifier()
pipeline = Pipeline(steps=[
    ('TfidfVectorize', TfidfVectorizer()),
    ('randomforest', randomforest)
])

# 6.1 Train the pipeline
pipeline.fit(x_train, y_train)

# Step 7: Evaluate the Pipeline
# Training accuracy
train_accuracy = pipeline.score(x_train, y_train)
print(f"Training Accuracy: {train_accuracy:.4f}")

# Testing accuracy
test_accuracy = pipeline.score(x_test, y_test)
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred = pipeline.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
from sklearn.metrics import classification_report
classification_re = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_re)

# Additional Metrics: Precision, Recall, and F1-Score for each class
from sklearn.metrics import precision_score, recall_score, f1_score

# Precision, Recall, F1-score for each class (emotion)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

print(f"Precision for each class: {precision}")
print(f"Recall for each class: {recall}")
print(f"F1 Score for each class: {f1}")

# If you want the overall average metrics:
precision_avg = precision_score(y_test, y_pred, average='weighted')
recall_avg = recall_score(y_test, y_pred, average='weighted')
f1_avg = f1_score(y_test, y_pred, average='weighted')

print(f"Average Precision: {precision_avg:.4f}")
print(f"Average Recall: {recall_avg:.4f}")
print(f"Average F1 Score: {f1_avg:.4f}")
