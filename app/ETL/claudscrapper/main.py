# main.py
from reddit_scraper import RedditScraper

def main():
    """Main function to run the scraper with user configuration."""
    print("Do you want to include comments? (yes/no):")
    include_comments = input().strip().lower() == 'yes'
    
    if include_comments:
        print("Enter maximum number of comment trees to expand (default is 10):")
        comment_input = input().strip()
        comment_limit = int(comment_input) if comment_input else 10
    else:
        comment_limit = 0
    
    scraper = RedditScraper(include_comments=include_comments, 
                           comment_limit=comment_limit)
    scraper.scrape_subreddits()

if __name__ == "__main__":
    main()