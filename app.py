import os
import logging
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key-for-development")

# Configure Google's Gemini API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Set up the Gemini model
def configure_genai():
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        return True
    else:
        logging.error("GOOGLE_API_KEY not found in environment variables")
        return False

def get_available_models():
    """Get a list of available models"""
    try:
        models = list(genai.list_models())
        logging.debug(f"Available models: {[m.name for m in models]}")
        return models
    except Exception as e:
        logging.error(f"Could not list models: {str(e)}")
        return []

def get_gemini_response(prompt):
    try:
        if not configure_genai():
            return "Error: API key not configured. Please set the GOOGLE_API_KEY environment variable."
        
        # Get available models
        models = get_available_models()
        model_found = False
        
        # Try different model names in order of preference
        model_names = [
            "models/gemini-1.5-pro",
            "models/gemini-pro",
            "models/gemini-1.0-pro",
            "gemini-1.5-pro",
            "gemini-pro"
        ]
        
        # If we have a list of models, try to find one that matches our preferences
        if models:
            for name in model_names:
                for model in models:
                    if name in model.name:
                        model_name = model.name
                        logging.debug(f"Found matching model: {model_name}")
                        model_found = True
                        break
                if model_found:
                    break
        
        # If no model was found, just try the first in our preference list
        if not model_found:
            model_name = model_names[0]
            logging.debug(f"No model found, defaulting to: {model_name}")
        
        # Initialize model
        logging.debug(f"Attempting to use model: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        # Generate content
        response = model.generate_content(prompt)
        
        # Return the response text
        return response.text
    except Exception as e:
        logging.error(f"Error getting response from Gemini: {str(e)}")
        return f"Error communicating with Gemini API: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Get response from Gemini
    bot_response = get_gemini_response(user_message)
    
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    # Run the app on port 5000 and bind to all interfaces
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
