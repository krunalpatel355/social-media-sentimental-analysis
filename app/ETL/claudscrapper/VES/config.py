import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGODB_URI = os.getenv('MONGODB_URI', "mongodb+srv://beau:bngeFBqJJoEWqRNd@cluster0.svcxhgj.mongodb.net/?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true")
    # You need to replace this with your valid HuggingFace token
    HF_TOKEN = os.getenv('HF_TOKEN', "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")  
    SUBREDDITS_FILE = os.getenv('SUBREDDITS_FILE', "subreddits.txt")
    EMBEDDING_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"