import os

def create_folder_structure():
    folders = [
        "static",
        "templates",
        "modules"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("Folders created successfully.")

def create_files():
    files_content = {
        "app.py": '''from flask import Flask
from routes import setup_routes

app = Flask(__name__)

# Setting up routes
setup_routes(app)

if __name__ == "__main__":
    app.run(debug=True)
''',

        "routes.py": '''from flask import render_template, request, jsonify
from modules.data_fetcher import fetch_data
from modules.data_processing import process_data

def setup_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/search', methods=['POST'])
    def search():
        keyword = request.form.get('keyword')
        data = fetch_data(keyword)
        processed_data = process_data(data)
        return jsonify(processed_data)
''',

        "modules/data_fetcher.py": '''import requests

def fetch_data(keyword):
    # Example: fetch data from Reddit API (placeholder)
    # Replace with actual logic
    response = requests.get(f"https://www.reddit.com/search.json?q={keyword}")
    return response.json()
''',

        "modules/data_processing.py": '''from sklearn.feature_extraction.text import TfidfVectorizer
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
''',

        "modules/utils.py": '''def clean_text(text):
    # Function to clean text data
    return text.strip().lower()
''',

        "templates/index.html": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search App</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1>What can I help with?</h1>
        <form id="search-form">
            <input type="text" name="keyword" placeholder="Type your query here" required>
            <button type="submit">Search</button>
        </form>
        <div id="results"></div>
    </div>

    <script>
        document.getElementById("search-form").onsubmit = async function(e) {
            e.preventDefault();
            const keyword = document.querySelector('input[name="keyword"]').value;
            const response = await fetch('/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ keyword })
            });
            const results = await response.json();
            document.getElementById("results").innerHTML = results.map(
                item => `<p>${item.title} - Similarity: ${item.similarity}</p>`
            ).join('');
        };
    </script>
</body>
</html>
''',

        "static/style.css": '''body {
    font-family: Arial, sans-serif;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
    background-color: #f8f9fa;
}

.container {
    text-align: center;
    max-width: 500px;
    padding: 20px;
    border: 1px solid #ddd;
    border-radius: 8px;
    background-color: #fff;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

h1 {
    font-size: 24px;
    margin-bottom: 20px;
}

input[type="text"] {
    width: 100%;
    padding: 10px;
    margin-bottom: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
}

button {
    padding: 10px 20px;
    color: #fff;
    background-color: #007bff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

button:hover {
    background-color: #0056b3;
}

#results {
    margin-top: 20px;
    text-align: left;
}
'''
    }

    for filepath, content in files_content.items():
        with open(filepath, "w") as f:
            f.write(content)
    print("Files created successfully.")

if __name__ == "__main__":
    create_folder_structure()
    create_files()
