# This module provides the main routes for the application's web interface.
# It handles the primary page rendering and user interactions.

from flask import Blueprint, render_template
from . import db  # Import the globally initialized db
from .models import User  # Import the User model

# Create a Blueprint for main application routes
main = Blueprint('main', __name__)

@main.route('/')
def index():
    """
    Render the main index page of the application.
    
    This endpoint serves as the entry point to the web interface. It retrieves
    all users from the database and passes them to the template for rendering.
    
    Returns:
        Rendered template 'index.html' with the following context:
        - users: List of all User objects from the database
    
    Note:
        Database operations are performed within the request context to ensure
        proper connection handling and transaction management.
    """
    # Example of using the database.
    # It's good practice to do database operations within a request context.
    users = User.query.all()
    return render_template('index.html', users=users)
