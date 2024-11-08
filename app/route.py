from flask import Blueprint, jsonify

# Define a blueprint for the app's routes
app = Blueprint('app', __name__)

@app.route('/')
def home():
    return "Welcome to the Flask App!"

 