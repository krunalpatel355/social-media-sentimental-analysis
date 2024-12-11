from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import sys
from config.settings import Config
from ETL import RedditScraper, ScraperConfig

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Initialize the DAG
dag = DAG(
    'reddit_scraper',
    default_args=default_args,
    description='A DAG to scrape Reddit data and store it in MongoDB',
    schedule_interval='@daily',  # Run daily, adjust as needed
    start_date=datetime(2023, 12, 1),
    catchup=False,
)

def scrape_reddit(**kwargs):
    """Airflow task to scrape Reddit data."""
    # Scraper Configuration
    config = ScraperConfig(
        subreddits=['politics', 'ukpolitics'],  # Customize your subreddits
        post_limit=10,
        include_comments=True,
        comment_limit=5,
    )

    # Initialize and run scraper
    scraper = RedditScraper(config)
    scraper.scrape_subreddits()

# Define Airflow task
scraper_task = PythonOperator(
    task_id='scrape_reddit_task',
    python_callable=scrape_reddit,
    provide_context=True,
    dag=dag,
)

# Define task dependencies
scraper_task
