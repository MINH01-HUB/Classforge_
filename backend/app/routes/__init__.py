# This module initializes and registers all route blueprints for the Flask application.
# Each blueprint represents a group of related endpoints that handle specific functionality.

from .csv_route import main as csv_blueprint
from .neo4j_route import graph as neo4j_blueprint
from .graph_route import graph_route as graph_blueprint  # ✅ Graph data endpoint
from .gat_route import gat as gat_blueprint                # ✅ GAT model endpoint
from .class_students_route import class_students_bp


def register_routes(app):
    """
    Register all route blueprints with the Flask application.
    
    This function registers the following blueprints:
    - csv_blueprint: Handles CSV file uploads and data processing
    - neo4j_blueprint: Manages Neo4j database operations
    - graph_blueprint: Provides graph data visualization endpoints
    - gat_blueprint: Handles Graph Attention Network model operations
    - class_students_bp: Manages classroom and student assignments
    
    Args:
        app: Flask application instance
    """
    app.register_blueprint(csv_blueprint)
    app.register_blueprint(neo4j_blueprint)
    app.register_blueprint(graph_blueprint)
    app.register_blueprint(gat_blueprint)           # ✅ Adds /run-gat route
    app.register_blueprint(class_students_bp)

