# server/app.py

import sys
import os

# allow imports from root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app