import os
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Import and register Blueprints
from routes.prediction import prediction_bp
from routes.file_upload import file_upload_bp

app.register_blueprint(prediction_bp, url_prefix='/predict')
app.register_blueprint(file_upload_bp, url_prefix='/upload')

@app.route('/')
def home():
    return 'Welcome to the Property Price Prediction API', 200

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", 5000))
