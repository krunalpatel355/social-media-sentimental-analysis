#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import rcParams
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from matplotlib import rcParams
rcParams['font.family'] = 'DejaVu Sans'
import emoji  # For emoji handling
import neattext.functions as nfx
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from transformers import pipeline, AutoTokenizer
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer


# In[5]:


df = pd.read_csv(r"C:\Users\Hazel\Downloads\Sentiment Analysis files\emotion_dataset.csv")


# In[3]:


df.head()


# In[4]:


# Define the expected emotions
expected_emotions = ["joy", "sadness", "fear", "anger", "surprise", "neutral", "disgust", "shame"]

# Count values and ensure all expected emotions are included
emotion_counts = df['Emotion'].value_counts()
emotion_counts = emotion_counts.reindex(expected_emotions, fill_value=0)

# Display results
print("'Emotion Counts':")
print(emotion_counts)

# Display total rows in the dataset
total_rows = len(df)
print("\nTotal Rows in Dataset: {:,}".format(total_rows))



# Removing duplicates

# In[5]:


df = df.drop_duplicates(subset=['Emotion', 'Text'], keep='first')



# Showing distribution graphically

# In[6]:


# Sort emotion counts for better visualization
emotion_counts = df['Emotion'].value_counts()

# Plot with a more appealing style and color palette
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

ax = sns.barplot(
    x=emotion_counts.index,
    y=emotion_counts.values,
    palette="muted"
)

# Add count values on top of bars
for i, value in enumerate(emotion_counts.values):
    ax.text(i, value + (value * 0.01), f'{value:,}', ha='center', va='bottom', fontsize=10)

# Format the Y-axis numbers with commas
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

# Set titles and labels
plt.title("Emotion Counts", fontsize=18)
plt.xlabel("Emotion", fontsize=14)
plt.ylabel("Count", fontsize=14)

# Show plot
plt.tight_layout()
plt.show()


# In[7]:


# Filter the dataset to only include rows labeled as "Joy"
joy_texts = df[df['Emotion'] == 'joy']

# Display the number of "Joy" samples
print(f"Number of 'Joy' samples: {len(joy_texts)}\n")

# Display the texts labeled as "Joy"
print("Sample 'Joy' texts:")
print(joy_texts[['Text', 'Emotion']].head(20))  # Print the first 20 samples


# Data pre-processing as raw data cannot be used to pre train the model it first needs to be preprocessed and cleaned

# Neattext will clean unstructured text data, reducing noice in text, and avoid repetition of the same code for pre-processing

# In[6]:


df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)


# Stopwords removal

# In[7]:


df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)


# In[11]:


# Display Initial Outliers (Current State)
df['Word_Count'] = df['Text'].apply(lambda x: len(str(x).split()))  # Compute Word Count

# Visualize Word Count Distribution
plt.figure(figsize=(12, 6))
df.boxplot(column='Word_Count', by='Emotion', grid=False, rot=45)
plt.title('Word Count Distribution by Emotion (Initial)')
plt.suptitle('')
plt.xlabel('Emotion')
plt.ylabel('Word Count')
plt.show()


# In[12]:


outliers = df[df['Word_Count'] > df['Word_Count'].quantile(0.95)]
print(f"Number of outliers (Initial): {len(outliers)}")
print(outliers[['Text', 'Emotion', 'Word_Count']].head())


# Implement changes in handling the outliers

# In[8]:


# Handle Emojis
def replace_emojis(text):
    """
    Consolidates emojis into categories or converts them to descriptive text.
    """
    positive_emojis = ["😊", "😁", "😍"]
    negative_emojis = ["😢", "😭", "😔"]
    for emo in positive_emojis:
        text = text.replace(emo, "positive_emotion")
    for emo in negative_emojis:
        text = text.replace(emo, "negative_emotion")
    # Convert remaining emojis to descriptive text
    text = emoji.demojize(text)
    return text

df['Clean_Text'] = df['Text'].apply(replace_emojis)


# Applied removal of userhandles and stopwords again

# In[9]:


df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)


# Another way is to remove outliers based on IQR

# In[15]:


# Remove Outliers Based on IQR
q1 = df['Word_Count'].quantile(0.25)
q3 = df['Word_Count'].quantile(0.75)
iqr = q3 - q1
df = df[(df['Word_Count'] >= (q1 - 1.5 * iqr)) & (df['Word_Count'] <= (q3 + 1.5 * iqr))]


# Display New Outliers After Mitigation

# In[12]:


df['Word_Count'] = df['Clean_Text'].apply(lambda x: len(str(x).split()))  # Recalculate Word Count
# Visualize Word Count Distribution After Mitigation
plt.figure(figsize=(12, 6))
df.boxplot(column='Word_Count', by='Emotion', grid=False, rot=45)
plt.title('Word Count Distribution by Emotion (After Mitigation)')
plt.suptitle('')
plt.xlabel('Emotion')
plt.ylabel('Word Count')
plt.show()


# In[17]:


# Identify New Outliers
outliers = df[df['Word_Count'] > df['Word_Count'].quantile(0.95)]
print(f"Number of outliers (After Mitigation): {len(outliers)}")
#difference of 1,742 from original 


# Splitting the data for input and target variable <br>
# X - All other features aside from the <br>
# Y - Target variable

# In[10]:


x = df['Clean_Text']
y = df['Emotion']


# Train and Split splitting

# In[13]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


# Pipeline is used to make the process of training the model faster and automated.<br>
# It automate workflows such as model fitting and training, by doing this, we can reduce time in processing and reduces model errors. <br>
# CountVectorizer is provided by scikit-learn to transform a given text into a vector on the basis of the frequency(count) <br>
# of each word that occurs in the entire text so machine can easily understand.  <br>
# This is used when there are multiple text that we wish to convert into further analysis: It does the following: <br>
# Tokenization: Breaks text into individual words (tokens). <br>
# Vocabulary Creation: Builds a list of unique words across all documents. <br>
# Count Matrix: Creates a matrix (document-term matrix) where rows represent documents and columns represent word frequencies. <br>

# We will use the following: <br>
# SVM - to find the optimal hyperplane separation through training the linearly separable data with the SVM algo <br>
# Random Forest the idea is that the combination of outputs of mutually exclusive nodes will outperform any individual models which will then predict the output by the use of combining multiple trees (Learner) <br>
# Logistic Regression - to find the relation between features and probability of particular outcome

# ***Why Scaling is not applied***
# The reason scaling is not necessary when using CountVectorizer is tied to the nature of how text data is processed and how scaling applies to machine learning models.
# 
# What Does Scaling Do?
# Scaling is a preprocessing step where numerical features are normalized to a common scale, typically within a range like [0, 1] or with a mean of 0 and standard deviation of 1. This is crucial for machine learning algorithms that are sensitive to the magnitude of feature values, such as:
# 
# Gradient-based algorithms: Logistic regression, support vector machines (SVM), or neural networks.
# Distance-based algorithms: K-nearest neighbors (KNN), clustering, etc.
# Why Scaling is Not Needed for CountVectorizer
# Output of CountVectorizer is Sparse and Discrete:
# 
# CountVectorizer transforms text data into a document-term matrix, where each row corresponds to a document, and each column corresponds to a word's frequency in that document.
# These counts are discrete and inherently on a comparable scale (e.g., word counts are integers starting from 0).

# Undersampling done as there is huge amount of 'Joy'

# In[20]:


print("\nApplying Random UnderSampling to balance the training classes...")
undersampler = RandomUnderSampler(random_state=42)
x_train_resampled, y_train_resampled = undersampler.fit_resample(pd.DataFrame(x_train), y_train)

# Convert x_train_resampled back to a Series for compatibility
x_train_resampled = x_train_resampled.squeeze()

# Output class distribution after undersampling
print("\nClass distribution in the training set after undersampling:")
class_distribution = y_train_resampled.value_counts()
print(class_distribution)

# Display the total number of samples left after undersampling
print("\nTotal number of samples left in the training set after undersampling:")
print(class_distribution.sum())

# Display the number of features per sample
# (Number of unique words or tokens in the training set)
pipe_cv = Pipeline(steps=[
    ('cv', CountVectorizer())
])
x_train_vectorized = pipe_cv.fit_transform(x_train_resampled)
print(f"\nNumber of features (unique words) after CountVectorizer: {x_train_vectorized.shape[1]}")


# The dataset dropped really low, what are we going to do to enhance the number?
# Rule-Based Relabeling

# Steps done:
# Process Overview
# Define Keywords:
# 
# Four sets of keywords are predefined for emotions to relabel "joy":
# Neutral: ["okay", "fine", "alright", "normal"]
# Surprise: ["unexpected", "wow", "unbelievable"]
# Disgust: ["gross", "disgusting", "nasty", "repulsive", "sickening"]
# Shame: ["ashamed", "embarrassed", "guilty", "humiliated", "regret"]
# Rule-Based Relabeling Function:
# 
# The function rule_based_relabel:
# Converts the input text to lowercase for case-insensitive matching.
# If the original label is "joy", it:
# Checks if any of the keywords for Neutral, Surprise, Disgust, or Shame appear in the text.
# Relabels the text accordingly.
# If no keywords match, the original label ("joy") is retained.
# Apply Rules to Dataset:
# Adjust Dataset:
# 
# Updates the "Adjusted_Emotion" column with the new or original labels based on the rules.
# Results:
# 
# Compares the original (Emotion) and adjusted (Adjusted_Emotion) emotion distributions.
# Output:
# 
# Prepares the dataset for training by updating:
# x_train_adjusted: Contains the processed text.
# y_train_adjusted: Contains the updated labels.
# 

# In[15]:


# Keywords for rule-based relabeling
neutral_keywords = ["okay", "fine", "alright", "normal"]
surprise_keywords = ["unexpected", "wow", "unbelievable"]
disgust_keywords = ["gross", "disgusting", "nasty", "repulsive", "sickening"]
shame_keywords = ["ashamed", "embarrassed", "guilty", "humiliated", "regret"]

# Rule-Based Relabeling
def rule_based_relabel(text, current_label):
    """
    Adjust emotion labels based on predefined keyword rules.
    """
    text_lower = text.lower()
    if current_label == "joy":  # Apply rule-based relabeling only for "joy"
        if any(keyword in text_lower for keyword in neutral_keywords):
            return "neutral"
        elif any(keyword in text_lower for keyword in surprise_keywords):
            return "surprise"
        elif any(keyword in text_lower for keyword in disgust_keywords):
            return "disgust"
        elif any(keyword in text_lower for keyword in shame_keywords):
            return "shame"
    return current_label  # Return the original label if no rule matches

# Apply Rule-Based Relabeling
df['Adjusted_Emotion'] = df.apply(
    lambda row: rule_based_relabel(row['Text'], row['Emotion']), axis=1
)

# Display Results
print("\nOriginal Emotion Distribution:")
print(df['Emotion'].value_counts())
print("\nAdjusted Emotion Distribution:")
print(df['Adjusted_Emotion'].value_counts())

# Update training variables
x_train_adjusted = df['Text']
y_train_adjusted = df['Adjusted_Emotion']


# Applying undersampling as there is still massive difference in Joy

# In[16]:


print("\nApplying Random UnderSampling to balance the training classes...")

# Step 1: Use the updated training variables
undersampler = RandomUnderSampler(random_state=42)
x_train_resampled, y_train_resampled = undersampler.fit_resample(
    pd.DataFrame(x_train_adjusted), y_train_adjusted
)

# Convert x_train_resampled back to a Series for compatibility
x_train_resampled = x_train_resampled.squeeze()

# Output class distribution after undersampling
print("\nClass distribution in the training set after undersampling:")
class_distribution = y_train_resampled.value_counts()
print(class_distribution)

# Display the total number of samples left after undersampling
print("\nTotal number of samples left in the training set after undersampling:")
print(class_distribution.sum())

# Step 2: Vectorize text data with CountVectorizer
pipe_cv = Pipeline(steps=[
    ('cv', CountVectorizer(max_features=5000))  # Optional max_features parameter for dimensionality reduction
])
x_train_vectorized = pipe_cv.fit_transform(x_train_resampled)

# Display the number of features (unique words or tokens) in the training set
print(f"\nNumber of features (unique words) after CountVectorizer: {x_train_vectorized.shape[1]}")



# ***Logistic Regression***
# 
# 

# In[23]:


print("\n--- Logistic Regression ---")

# Define the pipeline
pipe_lr = Pipeline(steps=[
    ('cv', CountVectorizer(max_features=5000)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
])

# Define parameter grid for Logistic Regression
param_grid_lr = {
    'cv__max_features': [500, 1000, 5000],
    'lr__C': [0.1, 1, 10]
}

# Perform grid search for Logistic Regression
grid_lr = GridSearchCV(pipe_lr, param_grid_lr, scoring='accuracy', cv=3, n_jobs=-1)
grid_lr.fit(x_train_resampled, y_train_resampled)

# Display the best parameters
best_params = grid_lr.best_params_
print("\nBest Parameters for Logistic Regression:")
print(best_params)

# Training Results
lr_train_pred = grid_lr.best_estimator_.predict(x_train_resampled)
lr_train_accuracy = accuracy_score(y_train_resampled, lr_train_pred)
print(f"Training Accuracy of Logistic Regression: {lr_train_accuracy * 100:.2f}%")

# Test Results
lr_test_pred = grid_lr.best_estimator_.predict(x_test)
lr_test_accuracy = accuracy_score(y_test, lr_test_pred)
print(f"Test Accuracy of Logistic Regression: {lr_test_accuracy * 100:.2f}%")

# Confusion Matrix
cm_lr = confusion_matrix(y_test, lr_test_pred)
print("\nConfusion Matrix for Logistic Regression (Test):")
print(cm_lr)

# Classification Report
print("\nClassification Report for Logistic Regression (Test):")
print(classification_report(y_test, lr_test_pred))

# AUC Calculation
try:
    lr_pred_proba = grid_lr.best_estimator_.predict_proba(x_test)
    if len(set(y_test)) == 2:  # Binary Classification
        auc_lr = roc_auc_score(y_test, lr_pred_proba[:, 1])
        print(f"AUC (Logistic Regression - Binary): {auc_lr:.2f}")
    else:  # Multiclass Classification
        auc_lr = roc_auc_score(y_test, lr_pred_proba, multi_class='ovr')
        print(f"AUC (Logistic Regression - Multiclass): {auc_lr:.2f}")
except Exception as e:
    print("\nAUC cannot be calculated for Logistic Regression due to:", str(e))

# Final Results Summary
print("\n--- Final Results for Logistic Regression ---")
print(f"Best Parameters: {best_params}")
print(f"Training Accuracy: {lr_train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {lr_test_accuracy * 100:.2f}%")
if 'auc_lr' in locals():
    print(f"AUC: {auc_lr:.2f}")


# ***SVM***

# In[24]:


print("\n--- Support Vector Machine ---")

# Define the pipeline
pipe_svm = Pipeline(steps=[
    ('cv', CountVectorizer(max_features=5000)),
    ('svc', SVC(probability=True, random_state=42))
])

# Define parameter grid for SVM
param_grid_svm = {
    'cv__max_features': [500, 1000, 5000],
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf']
}

# Perform grid search for SVM
grid_svm = GridSearchCV(pipe_svm, param_grid_svm, scoring='accuracy', cv=3, n_jobs=-1)
grid_svm.fit(x_train_resampled, y_train_resampled)

# Display the best parameters
best_params_svm = grid_svm.best_params_
print("\nBest Parameters for SVM:")
print(best_params_svm)

# Training Results
svm_train_pred = grid_svm.best_estimator_.predict(x_train_resampled)
svm_train_accuracy = accuracy_score(y_train_resampled, svm_train_pred)
print(f"Training Accuracy of SVM: {svm_train_accuracy * 100:.2f}%")

# Test Results
svm_test_pred = grid_svm.best_estimator_.predict(x_test)
svm_test_accuracy = accuracy_score(y_test, svm_test_pred)
print(f"Test Accuracy of SVM: {svm_test_accuracy * 100:.2f}%")

# Confusion Matrix
cm_svm = confusion_matrix(y_test, svm_test_pred)
print("\nConfusion Matrix for SVM (Test):")
print(cm_svm)

# Classification Report
print("\nClassification Report for SVM (Test):")
print(classification_report(y_test, svm_test_pred))

# AUC Calculation
try:
    svm_pred_proba = grid_svm.best_estimator_.predict_proba(x_test)
    if len(set(y_test)) == 2:  # Binary Classification
        auc_svm = roc_auc_score(y_test, svm_pred_proba[:, 1])
        print(f"AUC (SVM - Binary): {auc_svm:.2f}")
    else:  # Multiclass Classification
        auc_svm = roc_auc_score(y_test, svm_pred_proba, multi_class='ovr')
        print(f"AUC (SVM - Multiclass): {auc_svm:.2f}")
except Exception as e:
    print("\nAUC cannot be calculated for SVM due to:", str(e))

# Final Results Summary
print("\n--- Final Results for SVM ---")
print(f"Best Parameters: {best_params_svm}")
print(f"Training Accuracy: {svm_train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {svm_test_accuracy * 100:.2f}%")
if 'auc_svm' in locals():
    print(f"AUC: {auc_svm:.2f}")


# In[ ]:


print("\n--- Random Forest ---")

# Define the pipeline
pipe_rf = Pipeline(steps=[
    ('cv', CountVectorizer(max_features=5000)),
    ('rf', RandomForestClassifier(random_state=42))
])

# Define parameter grid for Random Forest
param_grid_rf = {
    'cv__max_features': [500, 1000, 5000],
    'rf__n_estimators': [100, 200, 500],
    'rf__max_depth': [10, 20, None]
}

# Perform grid search for Random Forest
grid_rf = GridSearchCV(pipe_rf, param_grid_rf, scoring='accuracy', cv=3, n_jobs=-1)
grid_rf.fit(x_train_resampled, y_train_resampled)

# Display the best parameters
best_params_rf = grid_rf.best_params_
print("\nBest Parameters for Random Forest:")
print(best_params_rf)

# Training Results
rf_train_pred = grid_rf.best_estimator_.predict(x_train_resampled)
rf_train_accuracy = accuracy_score(y_train_resampled, rf_train_pred)
print(f"Training Accuracy of Random Forest: {rf_train_accuracy * 100:.2f}%")

# Test Results
rf_test_pred = grid_rf.best_estimator_.predict(x_test)
rf_test_accuracy = accuracy_score(y_test, rf_test_pred)
print(f"Test Accuracy of Random Forest: {rf_test_accuracy * 100:.2f}%")

# Confusion Matrix
cm_rf = confusion_matrix(y_test, rf_test_pred)
print("\nConfusion Matrix for Random Forest (Test):")
print(cm_rf)

# Classification Report
print("\nClassification Report for Random Forest (Test):")
print(classification_report(y_test, rf_test_pred))

# AUC Calculation
try:
    rf_pred_proba = grid_rf.best_estimator_.predict_proba(x_test)
    if len(set(y_test)) == 2:  # Binary Classification
        auc_rf = roc_auc_score(y_test, rf_pred_proba[:, 1])
        print(f"AUC (Random Forest - Binary): {auc_rf:.2f}")
    else:  # Multiclass Classification
        auc_rf = roc_auc_score(y_test, rf_pred_proba, multi_class='ovr')
        print(f"AUC (Random Forest - Multiclass): {auc_rf:.2f}")
except Exception as e:
    print("\nAUC cannot be calculated for Random Forest due to:", str(e))

# Final Results Summary
print("\n--- Final Results for Random Forest ---")
print(f"Best Parameters: {best_params_rf}")
print(f"Training Accuracy: {rf_train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {rf_test_accuracy * 100:.2f}%")
if 'auc_rf' in locals():
    print(f"AUC: {auc_rf:.2f}")


# Choose model and check input

# In[ ]:


# Function for user input prediction
def predict_emotion(user_input, trained_model):
    """
    Predicts the emotion for the given user input using the trained model.

    Args:
        user_input (str): The text input from the user.
        trained_model (Pipeline): The trained pipeline model for prediction.

    Returns:
        str: Predicted emotion label.
    """
    # Predict emotion using the model pipeline
    predicted_label = trained_model.predict([user_input])[0]
    return predicted_label

# Example Usage: Allow user to select a model for prediction
print("\n--- Choose a Model for Prediction ---")

# Store trained models in a dictionary
models = {
    "logistic_regression": grid_lr.best_estimator_,
    "svm": grid_svm.best_estimator_,
    "random_forest": grid_rf.best_estimator_
}

# Prompt the user to select a model
chosen_model_name = input("Choose a model (logistic_regression, svm, random_forest): ").lower()

# Validate user input and select the corresponding model
if chosen_model_name in models:
    chosen_model = models[chosen_model_name]
    print(f"You selected: {chosen_model_name}")
else:
    print("Invalid model selection. Defaulting to Logistic Regression.")
    chosen_model = grid_lr.best_estimator_

# User Input Prediction Loop
while True:
    user_text = input("\nEnter text to predict the emotion (or type 'exit' to quit): ")
    if user_text.lower() == 'exit':
        print("Exiting...")
        break

    # Predict emotion
    predicted_emotion = predict_emotion(user_text, chosen_model)
    print(f"Predicted Emotion: {predicted_emotion}")


# In[ ]:


import joblib 
pipeline_file = open("text_emotion.pkl","wb") #write binary mode
joblib.dump(pipe_rf,pipeline_file)
pipeline_file.close


# In[ ]:




