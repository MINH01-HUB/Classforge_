# This module provides endpoints for retrieving student information by class.
# It allows querying all students or filtering by specific classroom assignments.

from flask import Blueprint, request, jsonify
from app.services.neo4j_service import Neo4jService

# Create a Blueprint for class-student related routes
class_students_bp = Blueprint('class_students', __name__)
neo4j_service = Neo4jService()

@class_students_bp.route('/api/class-students', methods=['GET'])
def get_class_students():
    """
    Retrieve students based on class assignment.
    
    This endpoint supports two modes:
    1. Get all students (when class='all')
    2. Get students from a specific class (when class=<class_name>)
    
    Query Parameters:
        class (str): Class name to filter by, or 'all' for all students
                    Default: 'all'
    
    Returns:
        JSON response containing:
        - Success (200): List of student objects with at least 'id' and 'group' fields
        - Error (500): Error message if the query fails
    
    Example Response:
        [
            {
                "id": "student1",
                "group": "Class_1",
                "achievement": 85.5,
                "wellbeing": 0.7
            },
            ...
        ]
    """
    class_name = request.args.get('class', 'all')
    try:
        if class_name == 'all':
            students = neo4j_service.get_all_students()
        else:
            students = neo4j_service.get_students_by_class(class_name)
        # Each student should have at least 'id' and 'group' (class)
        return jsonify(students), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500 