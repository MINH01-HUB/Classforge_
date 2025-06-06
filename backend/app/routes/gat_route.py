# This module provides endpoints for executing the Graph Attention Network (GAT) model.
# It handles the model execution requests from the frontend and manages the allocation
# of students to classrooms based on various weighted factors.

from flask import Blueprint, request, jsonify
from app.services.run_gat_and_export import run_gat_and_export  # Use updated execution file

# Create a Blueprint for GAT-related routes
gat = Blueprint("gat", __name__)

# --------------------------------------------
# POST /run-gat
# Description:
#   Executes the GAT model using preprocessed data from DB.
#   Accepts frontend slider weights to guide allocation logic.
# --------------------------------------------
@gat.route("/run-gat", methods=["POST"])
def run_gat():
    """
    Execute the GAT model for classroom allocation.
    
    This endpoint processes a request to run the Graph Attention Network model,
    which analyzes student relationships and generates optimal classroom assignments.
    The model takes into account various factors that can be weighted using sliders
    from the frontend interface.
    
    Request Body:
        {
            "sliders": {
                "academic_weight": float,  # Weight for academic performance
                "wellbeing_weight": float, # Weight for student wellbeing
                "friendship_weight": float,# Weight for maintaining friendships
                "conflict_weight": float,  # Weight for avoiding conflicts
                ... (other optional weights)
            }
        }
    
    Returns:
        JSON response containing:
        - Success (200): {
            "status": "success",
            "message": "GAT model ran successfully.",
            "result": {
                "neo4j_updated": bool,     # Whether Neo4j was updated
                "d3_file_path": str,       # Path to the generated D3 visualization file
                "total_students": int,     # Number of students processed
                "total_relationships": int,# Number of relationships in the graph
                "status": str             # Additional status information
            }
          }
        - Error (500): {
            "status": "error",
            "message": str  # Error details
          }
    
    Note:
        The model execution is resource-intensive and may take some time to complete.
        The frontend should handle this appropriately with loading indicators.
    """
    try:
        # 🎛️ Get slider weights from frontend (can be empty)
        sliders = request.json.get("sliders", {})
        print("📊 Slider Weights Received:", sliders)

        # 🚀 Run the GAT pipeline (DB-driven)
        result = run_gat_and_export(sliders)

        # ✅ Return success response with metadata
        return jsonify({
            "status": "success",
            "message": "GAT model ran successfully.",
            "result": result
        }), 200

    except Exception as e:
        # ❌ Handle and return error
        print("❗Error in /run-gat:", str(e))
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
