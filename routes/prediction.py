import json
import re
import logging
import openai
from flask import Blueprint, request, jsonify
from model import predict_property_value, feature_columns, prediction_model, scaler
from embeddings import model, knowledge_base, embeddings, index
from config import OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY
client = openai

prediction_bp = Blueprint('prediction', __name__)

def extract_property_details(user_input):
    area_match = re.search(r'(\d+)\s*area', user_input)
    bedrooms_match = re.search(r'(\d+)\s*bedrooms', user_input)
    bathrooms_match = re.search(r'(\d+)\s*bathrooms', user_input)
    location_match = re.search(r'in\s+(\w+)', user_input)
    year_built_match = re.search(r'built\s+in\s+(\d+)', user_input)
    property_type_match = re.search(r'property\s+type\s+is\s+(\w+)', user_input)
    
    return {
        'area': int(area_match.group(1)) if area_match else None,
        'bedrooms': int(bedrooms_match.group(1)) if bedrooms_match else None,
        'bathrooms': int(bathrooms_match.group(1)) if bathrooms_match else None,
        'location': location_match.group(1) if location_match else None,
        'year_built': int(year_built_match.group(1)) if year_built_match else None,
        'property_type': property_type_match.group(1) if property_type_match else None,
    }

def find_similar_response(user_input, embeddings, knowledge_base):
    query_embedding = model.encode([user_input])
    D, I = index.search(query_embedding, k=1)
    return knowledge_base[I[0][0]]

@prediction_bp.route('/', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    try:
        user_input = request.json.get('message', '')
        logger.debug(f"Received user input: {user_input}")
        
        details = extract_property_details(user_input)
        logger.debug(f"Extracted details: {details}")

        if all(details.values()):
            logger.debug("All required details found in the input")
            input_data = [
                details['area'], 
                details['bedrooms'], 
                details['bathrooms'], 
                details['year_built'], 
                1 if details['location'] == 'Location_A' else 0,  # Example encoding
                0,  # Example encoding
                1 if details['property_type'] == 'House' else 0,  # Example encoding
                0  # Example encoding
            ]
            logger.debug(f"Input data for prediction: {input_data}")
            predicted_price = predict_property_value(prediction_model, scaler, feature_columns, input_data)
            logger.debug(f"Predicted price: {predicted_price}")

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"The predicted price for the house is {predicted_price}."},
                ]
            )

            chat_response = response.choices[0].message.content
            logger.debug(f"Chat response: {chat_response}")

            return jsonify({"predicted_price": predicted_price, "response": chat_response})

        else:
            logger.debug("Not all required details found, treating as a general query")
            similar_response = find_similar_response(user_input, embeddings, knowledge_base)
            logger.debug(f"Similar response from knowledge base: {similar_response}")

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": similar_response},
                ]
            )

            chat_response = response.choices[0].message.content
            logger.debug(f"Chat response: {chat_response}")

            return jsonify({"response": chat_response})

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500
