"""
setup.py — Minimal setup script for editable installs.

The actual project metadata lives in pyproject.toml.
This file exists only for backward compatibility with older tools
that don't yet fully support PEP 517/518.

Usage:
    pip install -e .          # Editable install
    pip install -e ".[dev]"   # With dev extras
"""

from setuptools import setup

# All configuration is in pyproject.toml.
# This file is intentionally minimal.
setup()
