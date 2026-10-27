"""
conftest.py — pytest configuration for HAR project.
Adds the project src/ to the Python path for all tests.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))
