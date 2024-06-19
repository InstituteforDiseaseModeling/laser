import os
import sys

# Add the path to the 'utils' directory so Python can find its modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

# Now import the modules from 'utils'
from .utils import folium_animate_from_sqlite

