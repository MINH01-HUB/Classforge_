# This module provides endpoints for handling CSV file uploads and processing.
# It manages the secure upload of student data files and their integration into the system.

from flask import Blueprint, request, redirect, render_template, flash, jsonify
from werkzeug.utils import secure_filename
import os

from app.utils.csv_handler import upload_raw_csv

# Create a Blueprint for CSV-related routes
main = Blueprint("main", __name__)

# Configuration for file uploads
UPLOAD_FOLDER = "uploads"  # Directory where uploaded files are temporarily stored
ALLOWED_EXTENSIONS = {"csv"}  # Only CSV files are allowed

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename (str): The name of the file to check
        
    Returns:
        bool: True if the file extension is allowed, False otherwise
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route("/upload-csv", methods=["GET", "POST"])
def upload_csv():
    """
    Handle CSV file uploads and processing.
    
    This endpoint supports two operations:
    1. GET: Renders the upload form
    2. POST: Processes the uploaded CSV file
    
    For POST requests:
    - Validates the uploaded file
    - Securely saves it to the upload folder
    - Processes the CSV data using the csv_handler
    - Cleans up the temporary file
    - Returns success/error status
    
    Returns:
        For GET:
            Rendered template 'upload_csv.html'
        
        For POST:
            JSON response containing:
            - Success (200): {
                "status": "success",
                "message": "CSV uploaded successfully..."
              }
            - Error (400): {
                "status": "error",
                "message": "No file provided" or "Invalid file type..."
              }
            - Error (500): {
                "status": "error",
                "message": "Error uploading CSV: <error details>"
              }
    """
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return jsonify({"status": "error", "message": "No file provided"}), 400
            
        if not allowed_file(file.filename):
            return jsonify({"status": "error", "message": "Invalid file type. Please upload a CSV file"}), 400

        # Securely save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Process the CSV file
            upload_raw_csv(filepath)
            # Clean up the uploaded file
            os.remove(filepath)
            return jsonify({
                "status": "success",
                "message": "CSV uploaded successfully. Data saved to raw_student_data table."
            }), 200

        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                "status": "error",
                "message": f"Error uploading CSV: {str(e)}"
            }), 500

    # For GET requests, render the upload form
    return render_template("upload_csv.html")

