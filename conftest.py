"""Root conftest.py — ensures project root is on sys.path for pytest."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
