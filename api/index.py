"""
Vercel Serverless Function Entry Point for Paddy Leaf Disease Detection & Research Dashboard.
Exports the WSGI Flask application object `app`.
"""

import os
import sys
from pathlib import Path

# Ensure project root is at the head of sys.path for serverless module resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask_app.app import create_app

# Instantiate Flask application
app = create_app()

# Expose WSGI handler for Vercel
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
