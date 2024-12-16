# ML Flask Tip Prediction App

## Project Overview
This is a machine learning Flask application that predicts restaurant tips based on various features.

## Local Development
1. Clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Train the model: `python model.py`
5. Run the Flask app: `flask run`

## Docker Deployment
1. Build the Docker image: 
   ```
   docker build -t ml-flask-app .
   ```
2. Run the Docker container:
   ```
   docker run -p 5000:5000 ml-flask-app
   ```

## Deployment on Render
1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set the build command: `pip install -r requirements.txt`
4. Set the start command: `gunicorn app:create_app()`
5. Set environment variables if needed

## API Endpoints
- `/predict` (POST): Predict tip amount
- `/health` (GET): Health check endpoint

## Testing
Run tests using: `python -m pytest tests/`