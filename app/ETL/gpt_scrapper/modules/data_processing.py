from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def process_data(data):
    # Example: Extract and process post content
    posts = [post['data']['title'] for post in data['data']['children']]
    # Perform vector search
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(posts)
    similarity_matrix = cosine_similarity(vectors)

    # Process and return results
    results = [{"title": post, "similarity": sim} for post, sim in zip(posts, similarity_matrix[0])]
    return results
