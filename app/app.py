# app.py
from flask import Flask, render_template, request, jsonify
from datetime import datetime, UTC
import random  # For demo dashboard data
from ves import VES
from search import search 
from typing import List, Optional, Dict, Union
from dataclasses import dataclass, asdict
from dashboard import DASHBOARD
import pandas as pd
import numpy as np 
import plotly.express as px

app = Flask(__name__)


@dataclass
class SearchParameters:
    """Class to hold search parameters"""
    subreddits: List[str]
    from_time: Optional[datetime] = None
    to_time: Optional[datetime] = None
    sort_types: List[str] = None
    post_limit: Optional[int] = None
    include_comments: bool = False
    search_text: str = ""
    comment_limit: int = 10



@app.route('/')
def index():
    return render_template('index.html', active_page='search')

@app.route('/get_initial_data', methods=['POST'])
def get_initial_data():
    data = request.get_json()
    user_input = data.get('text') 
    ves_instance = VES(user_input)
    result_data = ves_instance.vector_search()

    return jsonify(result_data)

@app.route('/dashboard')
def dashboard():
    # Generate some sample data for the dashboard
    monthly_data = [random.randint(1000, 5000) for _ in range(12)]
    performance_data = [random.randint(60, 100) for _ in range(6)]
    status_data = {
        "Active": random.randint(100, 200),
        "Pending": random.randint(20, 50),
        "Completed": random.randint(300, 400)
    }

    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10, 20, 25, 30, 40],
        "size": [50, 60, 70, 80, 90],
        "color": [1, 2, 3, 4, 5]
    })

    # Create a Plotly scatter plot
    fig = px.scatter(df, x="x", y="y", size="size", color="color", title="Interactive Scatter Plot")
    
    # Generate the plot's HTML div
    graph_html = fig.to_html(full_html=False)
    

    return render_template('dashboard.html', 
                         active_page='dashboard',
                         monthly_data=monthly_data,
                         performance_data=performance_data,
                         status_data=status_data,
                         graph_html=graph_html)


@app.route('/simple_search', methods=['POST'])
def simple_search():
    try:
        # Extract selected_buttons from request
        selected_buttons = request.json.get('selected_buttons', [])
        
        # Initialize the search class with the selected_buttons
        search_instance = search(selected_buttons)
        
        # Perform the simple search
        results = search_instance.perform_simple_search()
        print(results)
        print(type(results))
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/advanced_search', methods=['POST'])
def advanced_search():
    try:
        data = request.json
        # Map input fields to SearchParameters dataclass fields
        # params = {
        #     "subreddits": data.get('selected_buttons', []),  # Rename topics to subreddits
        #     "from_time": datetime.fromisoformat(data['from_time']) if 'from_time' in data else None,
        #     "to_time": datetime.fromisoformat(data['to_time']) if 'to_time' in data else None,
        #     "sort_types": data.get('post_types', []),  # Optional parameter
        #     "post_limit": int(data.get('post_limit', 0)),
        #     "include_comments": data.get('include_comments', 'no') == 'yes',
        #     "search_text": data.get('search_text', ""),
        # }

        # print(params)

        parameters = SearchParameters(
            subreddits=data.get('selected_buttons', []),
            from_time=datetime.fromisoformat(data['from_time']) if 'from_time' in data else None,  # Added timezone info
            to_time=datetime.fromisoformat(data['to_time']) if 'to_time' in data else None,
            sort_types=data.get('post_types', []),
            post_limit=int(data.get('post_limit', 0)),
            include_comments=data.get('include_comments', False) == True,
            search_text=data.get('search_text', ""),
            comment_limit=10
        )
        # # Initialize the search class with advanced parameters
        search_instance = search(parameters)
        results = search_instance.perform_advance_search()
        print(results)
        print(type(results))
        
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
