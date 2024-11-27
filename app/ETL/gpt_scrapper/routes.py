from flask import render_template, request, jsonify
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
