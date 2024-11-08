from flask import Flask

# Define a factory function to create the app
def create_app():
    app = Flask(__name__)

    # Import and register blueprints or routes
    from .route import app as route_app
    app.register_blueprint(route_app)

    return app

 