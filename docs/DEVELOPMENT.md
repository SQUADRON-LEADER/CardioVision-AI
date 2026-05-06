# Development Guide

## Setup
1. Create virtual environment: python -m venv .venv
2. Activate: .venv\Scripts\activate
3. Install: pip install -r backend/requirements.txt

## Running Locally
Start backend: python backend/app.py
Access at: http://localhost:5000

## Testing
Run tests: python -m pytest tests/

## Project Structure
- backend/: Flask API
- frontend/: Web UI
- training/: Model training notebooks
- tests/: Test files

## Contributing
Follow conventional commits:
- feat: new feature
- fix: bug fix
- docs: documentation
- refactor: code refactoring
- test: tests
- chore: maintenance
