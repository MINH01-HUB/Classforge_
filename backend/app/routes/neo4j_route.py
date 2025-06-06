# app/routes/neo4j_route.py

# This module provides endpoints for managing the Neo4j graph database connection and operations.
# It handles database status checks, graph resets, and other Neo4j-related operations.

from flask import Blueprint, jsonify
from neo4j import GraphDatabase
import os

# Create a Blueprint for Neo4j-related routes
graph = Blueprint("neo4j", __name__)

# Load Neo4j connection settings from environment or fallback
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")  # Default to local Neo4j instance
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")               # Default Neo4j username
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword") # Default password

# Initialize the Neo4j driver with connection settings
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

@graph.route("/neo4j/status", methods=["GET"])
def check_neo4j():
    """
    Check the connection status of the Neo4j database.
    
    This endpoint verifies the connection to Neo4j and returns the current
    number of nodes in the graph. It's useful for monitoring the database
    health and connection status.
    
    Returns:
        JSON response containing:
        - Success (200): {
            "status": "connected",
            "node_count": int  # Number of nodes in the graph
          }
        - Error (500): {
            "status": "error",
            "message": str  # Error details
          }
    
    Note:
        This endpoint performs a lightweight query to count nodes,
        making it suitable for health checks and monitoring.
    """
    try:
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS count")
            count = result.single()["count"]
            return jsonify({"status": "connected", "node_count": count})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@graph.route("/neo4j/reset", methods=["POST"])
def reset_graph():
    """
    Reset the Neo4j graph database by removing all nodes and relationships.
    
    This endpoint performs a complete reset of the graph database by:
    1. Detaching all relationships from nodes
    2. Deleting all nodes
    This operation is irreversible and should be used with caution.
    
    Returns:
        JSON response containing:
        - Success (200): {
            "message": "✅ Neo4j graph reset complete."
          }
        - Error (500): {
            "error": str  # Error details
          }
    
    Note:
        This is a destructive operation that will remove all data from the graph.
        It should only be used when a complete reset is necessary, such as:
        - Initial setup
        - Testing
        - Complete data refresh
    """
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        return jsonify({"message": "✅ Neo4j graph reset complete."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
