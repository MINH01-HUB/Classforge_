#version 3.0.0
# routes/graph_route.py
# This module provides endpoints for managing and retrieving graph data for the classroom visualization.
# It implements a robust system that prioritizes live Neo4j data but falls back to cached JSON when needed.
# The module handles both graph data retrieval and relationship management between students.

from flask import Blueprint, jsonify, request
from app.services.neo4j_service import neo4j_service, RELATIONSHIP_TYPES
import os
import json

# Create a Blueprint for graph-related endpoints
graph_route = Blueprint("graph_route", __name__)

# Name of the fallback JSON file to search for if Neo4j fails
FALLBACK_FILENAME = "classroom_graph_d3_for_frontend.json"

def find_fallback_json(start_dir="."):
    """
    Recursively search for the fallback JSON file starting from the given directory.
    
    This function implements a fallback mechanism to ensure graph data is always available,
    even when Neo4j is unavailable. It searches through the directory structure to find
    a cached version of the graph data.
    
    Args:
        start_dir (str): The directory to start searching from. Defaults to current directory.
    
    Returns:
        str or None: The full path to the fallback JSON file if found, None otherwise.
    """
    for root, dirs, files in os.walk(start_dir):
        if FALLBACK_FILENAME in files:
            return os.path.join(root, FALLBACK_FILENAME)
    return None

@graph_route.route("/get-graph", methods=["GET"])
def get_graph_data():
    """
    API endpoint to provide graph data for frontend visualization.
    
    This endpoint implements a two-tier data retrieval strategy:
    1. Primary: Attempts to fetch live graph data from Neo4j
    2. Fallback: If Neo4j is unavailable, retrieves cached data from JSON file
    
    The endpoint ensures data integrity by validating the presence of both nodes and links
    in the returned data structure.
    
    Returns:
        JSON response containing:
        - Success (200): Graph data with nodes and links
        - Error (500): Error message if both Neo4j and fallback fail
    
    Example Response:
        {
            "nodes": [
                {"id": "student1", "group": "Class_1", ...},
                ...
            ],
            "links": [
                {"source": "student1", "target": "student2", "type": "FRIENDS", ...},
                ...
            ]
        }
    """
    try:
        # Try to export graph data directly from Neo4j
        data = neo4j_service.export_d3_graph()
        if not data or not data.get("nodes") or not data.get("links"):
            raise Exception("No data returned from Neo4j")
        return jsonify(data)
    except Exception as e:
        # If Neo4j fails, attempt to use fallback JSON
        print(f"⚠️ Neo4j fetch failed, using fallback JSON: {e}")
        try:
            fallback_path = find_fallback_json()
            if fallback_path and os.path.exists(fallback_path):
                with open(fallback_path, "r") as f:
                    fallback_data = json.load(f)
                if not fallback_data or not fallback_data.get("nodes") or not fallback_data.get("links"):
                    raise Exception("Invalid fallback data format")
                return jsonify(fallback_data)
            else:
                raise Exception("Fallback JSON file not found")
        except Exception as fallback_error:
            print(f"⚠️ Fallback JSON failed: {fallback_error}")
            return jsonify({"error": "Graph data unavailable"}), 500

@graph_route.route("/relationship", methods=["POST"])
def create_relationship():
    """
    Create a new relationship between two students.
    
    This endpoint allows creating various types of relationships (e.g., friendships, conflicts)
    between students with an optional attention score.
    
    Request Body:
        {
            "student1_id": str,
            "student2_id": str,
            "relationship_type": str,
            "attention": float (optional, default: 0.5)
        }
    
    Returns:
        JSON response containing:
        - Success (200): Created relationship details
        - Error (400): Invalid input parameters
        - Error (500): Server error during creation
    """
    data = request.json
    try:
        result = neo4j_service.create_relationship(
            data["student1_id"],
            data["student2_id"],
            data["relationship_type"],
            data.get("attention", 0.5)
        )
        return jsonify({"success": True, "result": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Failed to create relationship"}), 500

@graph_route.route("/relationship", methods=["PUT"])
def update_relationship():
    """
    Update an existing relationship between two students.
    
    This endpoint allows modifying the type and attention score of an existing
    relationship between students.
    
    Request Body:
        {
            "student1_id": str,
            "student2_id": str,
            "relationship_type": str,
            "attention": float
        }
    
    Returns:
        JSON response containing:
        - Success (200): Updated relationship details
        - Error (500): Server error during update
    """
    data = request.json
    try:
        result = neo4j_service.update_relationship(
            data["student1_id"],
            data["student2_id"],
            data["relationship_type"],
            data["attention"]
        )
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": "Failed to update relationship"}), 500

@graph_route.route("/relationship", methods=["DELETE"])
def delete_relationship():
    """
    Delete a relationship between two students.
    
    This endpoint removes any existing relationship between the specified students.
    
    Request Body:
        {
            "student1_id": str,
            "student2_id": str
        }
    
    Returns:
        JSON response containing:
        - Success (200): Deletion confirmation
        - Error (500): Server error during deletion
    """
    data = request.json
    try:
        result = neo4j_service.delete_relationship(
            data["student1_id"],
            data["student2_id"]
        )
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": "Failed to delete relationship"}), 500

@graph_route.route("/relationship-types", methods=["GET"])
def get_relationship_types():
    """
    Get all available relationship types.
    
    This endpoint returns a list of all possible relationship types that can be
    created between students (e.g., FRIENDS, CONFLICT, etc.).
    
    Returns:
        JSON response containing:
        - Success (200): List of relationship types
    """
    return jsonify(list(RELATIONSHIP_TYPES.values()))

#Version 1.0.0 without decoding
# # routes/graph_route.py
# # This route provides a frontend-friendly graph data API.
# # If Neo4j is connected and available, it fetches live graph data.
# # If Neo4j is down or unavailable, it falls back to reading from a JSON file on disk.

# from flask import Blueprint, jsonify
# from app.services.neo4j_service import neo4j_service
# import os
# import json

# # Register the blueprint for graph-related endpoints
# graph_route = Blueprint("graph_route", __name__)

# # Name of the fallback JSON file to search for if Neo4j fails
# FALLBACK_FILENAME = "classroom_graph_d3_for_frontend.json"


# def find_fallback_json(start_dir="."):
#     """
#     Recursively search for the fallback JSON file starting from the given directory.
#     Returns the path if found, otherwise None.
#     """
#     for root, dirs, files in os.walk(start_dir):
#         if FALLBACK_FILENAME in files:
#             return os.path.join(root, FALLBACK_FILENAME)
#     return None


# @graph_route.route("/get-graph", methods=["GET"])
# def get_graph_data():
#     """
#     API endpoint to provide graph data for frontend visualization.
#     Tries to fetch from Neo4j first; falls back to local JSON file if necessary.
#     """
#     try:
#         # Try to export graph data directly from Neo4j
#         data = neo4j_service.export_d3_graph()
#         return jsonify(data)
#     except Exception as e:
#         # If Neo4j fails, attempt to use fallback JSON
#         print(f"⚠️ Neo4j fetch failed, using fallback JSON: {e}")
#         fallback_path = find_fallback_json()
#         if fallback_path and os.path.exists(fallback_path):
#             with open(fallback_path, "r") as f:
#                 fallback_data = json.load(f)
#             return jsonify(fallback_data)
#         else:
#             return jsonify({"error": "Graph data unavailable"}), 500
