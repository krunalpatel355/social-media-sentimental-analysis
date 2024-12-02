from flask import Flask, render_template, request, jsonify
from datetime import datetime, UTC
import random  # For demo dashboard data
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from typing import List, Optional, Dict, Union
from dataclasses import dataclass, asdict
from app.dashboard import DASHBOARD
from app.ves import VES
from app.search import search 
from app.config.settings import Config


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
    # Use search_id from a recent search or a default value
    search_id = "8cba419b-aa5b-4e80-8c96-70b520efdc1b"  # You might want to manage this dynamically
    
    # Initialize dashboard with search_id
    dashboard_instance = DASHBOARD(search_id)
    
    # Get sentiment analysis results
    mongo_uri = Config.MONGODB_URI  # Replace with your actual MongoDB URI
    analyzer = dashboard_instance.get_sentiment_analyzer(mongo_uri)
    
    # Sentiment Analysis
    sentiment_results = analyzer.perform_sentiment_analysis(search_id)
    
    # Segmentation Analysis
    segmentation_results = analyzer.perform_segmentation_analysis(search_id)

    # Prepare Sentiment Pie Chart
    sentiment_pie_data = [
        {'category': 'Positive', 'value': sentiment_results['overall']['positive_percentage']},
        {'category': 'Neutral', 'value': sentiment_results['overall']['neutral_percentage']},
        {'category': 'Negative', 'value': sentiment_results['overall']['negative_percentage']}
    ]
    
    pie_chart = go.Figure(data=[go.Pie(
        labels=[item['category'] for item in sentiment_pie_data],
        values=[item['value'] for item in sentiment_pie_data],
        hole=.3,
        marker_colors=['green', 'gray', 'red']
    )])
    pie_chart_html = pie_chart.to_html(full_html=False)

    # Prepare Cluster Visualization
    cluster_data = []
    for cluster in segmentation_results['clusters']:
        cluster_data.append({
            'cluster_id': cluster['cluster_id'],
            'size': cluster['size'],
            'top_terms': ', '.join(cluster['top_terms'][:5])
        })
    
    # Create Cluster Size Pie Chart
    cluster_pie_chart = go.Figure(data=[go.Pie(
        labels=[f"Cluster {c['cluster_id']}" for c in cluster_data],
        values=[c['size'] for c in cluster_data],
        hole=.3
    )])
    cluster_pie_chart_html = cluster_pie_chart.to_html(full_html=False)

    # Prepare Cluster Sunburst Chart (showing hierarchy)
    sunburst_data = []
    for cluster in segmentation_results['clusters']:
        sunburst_data.append(dict(
            ids=f"Cluster {cluster['cluster_id']}",
            labels=f"Cluster {cluster['cluster_id']}",
            parents="",
            values=cluster['size']
        ))
        # Add top terms as children
        for term in cluster['top_terms'][:3]:
            sunburst_data.append(dict(
                ids=f"Cluster {cluster['cluster_id']} - {term}",
                labels=term,
                parents=f"Cluster {cluster['cluster_id']}",
                values=1
            ))
    
    sunburst_chart = go.Figure(go.Sunburst(
        ids=[d['ids'] for d in sunburst_data],
        labels=[d['labels'] for d in sunburst_data],
        parents=[d['parents'] for d in sunburst_data],
        values=[d['values'] for d in sunburst_data]
    ))
    sunburst_chart_html = sunburst_chart.to_html(full_html=False)

    return render_template('dashboard.html', 
                         active_page='dashboard',
                         sentiment_results=sentiment_results,
                         segmentation_results=segmentation_results,
                         pie_chart_html=pie_chart_html,
                         cluster_pie_chart_html=cluster_pie_chart_html,
                         sunburst_chart_html=sunburst_chart_html)


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