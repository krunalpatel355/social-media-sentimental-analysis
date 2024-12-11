from pymongo import MongoClient
from datetime import datetime

    # MongoDB Atlas connection string (replace <username>, <password>, and <cluster_url> with your details)
MONGO_URI = "mongodb+srv://flask-app:SAVhq6YW1eW3Gqtp@cluster0.lu5p4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    # Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)

    # Define the database and collection
db = client["reddit_db"]
collection = db["search"]  # Replace "searches" with your actual collection name

try:
        # Query to find the latest document by timestamp
    latest_search = collection.find_one(sort=[("timestamp", -1)])

    if latest_search:
        search_id = latest_search.get("search_id")
        print("Latest search ID:", search_id)
    else:
        print("No search data found.")

except Exception as e:
    print("An error occurred while fetching data:", e)

finally:
        # Close the MongoDB connection
    client.close()