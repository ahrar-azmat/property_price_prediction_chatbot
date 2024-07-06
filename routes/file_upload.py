import os
import logging
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from utils.file_processing import extract_text_from_file

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

file_upload_bp = Blueprint('file_upload', __name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'xlsx', 'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@file_upload_bp.route('/', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            logger.debug(f"File saved to {file_path}")

            text = extract_text_from_file(file_path)
            logger.debug(f"Extracted text: {text}")

            # Here you would typically add the text to your vector database and handle embeddings

            return jsonify({"message": "File successfully uploaded and processed"}), 200
        else:
            return jsonify({"error": "File type not allowed"}), 400

    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        return jsonify({"error": str(e)}), 500
