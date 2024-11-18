# routes.py
from flask import render_template, jsonify, request, Blueprint

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/chat')
def chat():
    return render_template('chat.html')

@main.route('/settings')
def settings():
    return render_template('settings.html')
