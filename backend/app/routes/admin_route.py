# This module provides endpoints for the administrative interface of the application.
# It handles the rendering of the admin panel and related administrative functions.

from flask import Blueprint, render_template
main = Blueprint('main', __name__)


@main.route("/admin", methods=["GET"])
def admin_panel():
    """
    Render the administrative panel interface.
    
    This endpoint serves the admin panel page, which provides administrative
    controls and monitoring capabilities for the application. The admin panel
    is accessible only to authorized administrators.
    
    Returns:
        Rendered template 'admin.html' containing the administrative interface.
    
    Note:
        This endpoint should be protected by authentication middleware to ensure
        only authorized administrators can access it. The template should include
        appropriate access controls and administrative functions.
    """
    return render_template("admin.html")