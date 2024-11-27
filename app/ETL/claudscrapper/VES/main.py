from VES.config import Config
from VES.vector_search import VectorSearch
def ETL():
    try:
        vector_search = VectorSearch()
        
        if vector_search.needs_initialization():
            print("Initializing database with subreddit data...")
            vector_search.load_subreddits_from_file(Config.SUBREDDITS_FILE)
        else:
            print("Database already initialized, proceeding with vector search...")
        
        while True:
            query = input("\nEnter your search query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            results = vector_search.search_similar_subreddits(query)
            print(f"\nSimilar subreddits: {results}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return

