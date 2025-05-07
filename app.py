import os
import logging
import json
import requests
import datetime
import uuid
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key-for-development")

# Database configuration
# Handle the 'postgres://' vs 'postgresql://' issue if needed
database_url = os.environ.get('DATABASE_URL', 'sqlite:///vertex.db')
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Code snippet model
class CodeSnippet(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    code = db.Column(db.Text, nullable=False)
    language = db.Column(db.String(50), nullable=False)
    tags = db.Column(db.String(255), nullable=True)  # Comma-separated tags
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'code': self.code,
            'language': self.language,
            'tags': self.tags.split(',') if self.tags else [],
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

# Create database tables if they don't exist
with app.app_context():
    try:
        db.create_all()
        logging.info("Database tables created successfully")
    except Exception as e:
        logging.warning(f"Error creating tables, they may already exist: {str(e)}")
        pass

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
    """Get a list of available models that are text-only (no vision models)"""
    try:
        all_models = list(genai.list_models())
        logging.debug(f"All available models: {[m.name for m in all_models]}")
        
        # Filter out vision models to avoid errors for text-only tasks
        text_models = [m for m in all_models if 'vision' not in m.name.lower()]
        logging.debug(f"Filtered text-only models: {[m.name for m in text_models]}")
        
        return text_models
    except Exception as e:
        logging.error(f"Could not list models: {str(e)}")
        return []

def get_preferred_model():
    """Get the best available model based on our preferences"""
    try:
        # Get available models
        models = get_available_models()
        model_found = False
        
        # Try different model names in order of preference
        model_names = [
            "models/gemini-1.5-flash",  # Recommended by error message
            "models/gemini-1.5-flash-002",
            "models/gemini-1.5-flash-8b", # Even lighter model
            "models/gemini-2.0-flash-lite", # Another option
            "models/text-bison-001",  # Fallback to older model if needed
            "models/chat-bison-001"  # Another older model
        ]
        
        # If we have a list of models, try to find one that matches our preferences
        if models:
            for name in model_names:
                for model in models:
                    if name in model.name:
                        model_name = model.name
                        logging.debug(f"Found matching model: {model_name}")
                        return model_name
        
        # If no model was found, just return the first in our preference list
        return model_names[0]
    except Exception as e:
        logging.error(f"Error getting preferred model: {str(e)}")
        return "models/gemini-1.5-flash"  # Default fallback

def get_gemini_response(prompt, specific_model=None):
    """Get a response from a specific model or the best available model"""
    try:
        if not configure_genai():
            return "Error: API key not configured. Please set the GOOGLE_API_KEY environment variable."
        
        # Use specified model or get preferred model
        if specific_model:
            model_name = specific_model
        else:
            model_name = get_preferred_model()
            
        logging.debug(f"Attempting to use model: {model_name}")
        
        try:
            # Initialize model
            model = genai.GenerativeModel(model_name)
            
            # Configure generation parameters based on model type
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,  # Default temperature
                top_p=0.9,
                top_k=30
            )
            
            # Adjust parameters based on model
            if 'flash' in model_name.lower():
                # Flash models are optimized for speed, lower temperature for consistency
                generation_config.temperature = 0.6
            elif '1.5-pro' in model_name.lower() or 'ultra' in model_name.lower():
                # Pro/Ultra models can handle more creativity
                generation_config.temperature = 0.8
                generation_config.top_k = 40
            
            # Generate content with configured parameters
            response = model.generate_content(prompt, generation_config=generation_config)
            
            # Return the response text
            return response.text
        except Exception as e:
            # If the selected model fails, try a direct fallback
            logging.error(f"Error with model {model_name}: {str(e)}")
            fallback_model = "models/text-bison-001"
            logging.debug(f"Primary model failed, attempting fallback to: {fallback_model}")
            model = genai.GenerativeModel(fallback_model)
            response = model.generate_content(prompt)
            return response.text
            
    except Exception as e:
        logging.error(f"Error getting response from Vertex: {str(e)}")
        return f"Error communicating with Vertex assistant: {str(e)}"

def get_model_info():
    """Get information about available and upcoming AI models"""
    try:
        # Get the list of actual available models
        available_models = []
        
        try:
            if configure_genai():
                model_list = get_available_models()
                available_models = [m.name for m in model_list]
        except:
            # If we can't get the models, we'll just use the info without availability status
            pass
            
        # Comprehensive model information
        model_info = {
            "models/gemini-1.5-flash": {
                "name": "Gemini 1.5 Flash",
                "description": "Optimized for speed and efficiency, great for quick responses and high-throughput applications",
                "strengths": "Fast responses, lower latency, cost-efficient for common tasks",
                "best_for": "Chat applications, content moderation, classification tasks",
                "token_limit": 16384,
                "version": "1.5",
                "available": "models/gemini-1.5-flash" in available_models
            },
            "models/gemini-1.5-pro": {
                "name": "Gemini 1.5 Pro",
                "description": "Balanced performance with strong reasoning and text generation capabilities",
                "strengths": "General purpose, well-rounded capabilities for most tasks",
                "best_for": "Content generation, summarization, complex Q&A",
                "token_limit": 32768,
                "version": "1.5",
                "available": "models/gemini-1.5-pro" in available_models
            },
            "models/gemini-1.5-ultra": {
                "name": "Gemini 1.5 Ultra",
                "description": "Google's most capable model with exceptional reasoning and problem-solving",
                "strengths": "Complex reasoning, code generation, creative content, expert-level responses",
                "best_for": "Advanced problem-solving, code generation, complex analysis",
                "token_limit": 32768,
                "version": "1.5",
                "available": "models/gemini-1.5-ultra" in available_models
            },
            "models/gemini-pro-vision": {
                "name": "Gemini Pro Vision",
                "description": "Multimodal model that can understand both text and images",
                "strengths": "Image understanding and description, visual problem-solving",
                "best_for": "Image analysis, visual content generation, multimodal applications",
                "token_limit": 16384,
                "version": "1.0",
                "available": "models/gemini-pro-vision" in available_models
            },
            "text-bison": {
                "name": "PaLM 2 Text Bison",
                "description": "Older text generation model based on PaLM 2 architecture",
                "strengths": "Reliable text generation with good stability",
                "best_for": "Simple text generation tasks, basic Q&A",
                "token_limit": 8192,
                "version": "001",
                "available": "models/text-bison-001" in available_models
            }
        }
        
        return model_info
        
    except Exception as e:
        logging.error(f"Error getting model info: {str(e)}")
        # Return minimal info if error occurs
        return {
            "models/gemini-1.5-flash": {
                "name": "Gemini 1.5 Flash",
                "description": "Fast and efficient model for common tasks",
                "strengths": "Speed and efficiency",
                "best_for": "Quick responses",
                "token_limit": 16384,
                "version": "1.5",
                "available": False
            }
        }

@app.route('/')
def index():
    return render_template('home.html', active_page='home')

@app.route('/chat')
def chat_page():
    return render_template('chat.html', active_page='chat')

@app.route('/pricing')
def pricing_page():
    return render_template('pricing.html', active_page='pricing')

@app.route('/connect')
def connect_page():
    return render_template('connect.html', active_page='connect')

@app.route('/history')
def history_page():
    # Placeholder for chat history page
    return render_template('base.html', active_page='history')

@app.route('/settings')
def settings_page():
    return render_template('settings.html', active_page='settings')

@app.route('/help')
def help_page():
    # Placeholder for help page
    return render_template('base.html', active_page='help')

@app.route('/models')
def models_page():
    """Display the AI model comparison page"""
    # Get model information
    models = get_model_info()
    return render_template('models.html', active_page='models', models=models)
    
@app.route('/api/models/compare', methods=['POST'])
def compare_models():
    """API endpoint to compare responses from different AI models"""
    data = request.json
    prompt = data.get('prompt', '')
    models = data.get('models', [])
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    if not models or len(models) < 1:
        return jsonify({'error': 'At least one model must be selected'}), 400
    
    # Generate responses from each model
    responses = {}
    generation_times = {}
    
    for model_id in models:
        # Record generation time for performance comparison
        start_time = datetime.datetime.now()
        
        # Get response from model
        response = get_gemini_response(prompt, specific_model=model_id)
        
        # Calculate generation time
        end_time = datetime.datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        # Store response and generation time
        responses[model_id] = response
        generation_times[model_id] = round(generation_time, 2)
    
    return jsonify({
        'responses': responses,
        'generation_times': generation_times
    })

# Code snippet library routes
def get_user_recommendations(limit=6):
    """
    Generate personalized snippet recommendations based on:
    1. Most popular snippets (by view count)
    2. Recently added snippets
    3. Language preferences (based on user's viewed snippets)
    4. Tag preferences (based on user's viewed snippets)
    
    Returns a list of recommended snippet objects
    """
    recommendations = []
    try:
        # Get most viewed snippets (to be implemented when view tracking is added)
        # popular_snippets = CodeSnippet.query.order_by(CodeSnippet.view_count.desc()).limit(3).all()
        
        # For now, get newest snippets
        newest_snippets = CodeSnippet.query.order_by(CodeSnippet.created_at.desc()).limit(3).all()
        recommendations.extend(newest_snippets)
        
        # Get language preferences - we'll infer from the distribution of languages in the database
        language_counts = db.session.query(CodeSnippet.language, db.func.count(CodeSnippet.language))\
            .group_by(CodeSnippet.language)\
            .order_by(db.func.count(CodeSnippet.language).desc())\
            .all()
        
        if language_counts:
            # Get top language
            top_language = language_counts[0][0]
            
            # Get snippets in the most popular language, excluding already recommended
            already_recommended_ids = [s.id for s in recommendations]
            language_recommendations = CodeSnippet.query.filter(
                CodeSnippet.language == top_language,
                ~CodeSnippet.id.in_(already_recommended_ids)
            ).order_by(db.func.random()).limit(2).all()
            
            recommendations.extend(language_recommendations)
        
        # Get tag preferences by finding the most common tags
        all_tags = {}
        for snippet in CodeSnippet.query.all():
            if snippet.tags:
                for tag in snippet.tags.split(','):
                    tag = tag.strip()
                    all_tags[tag] = all_tags.get(tag, 0) + 1
        
        if all_tags:
            # Get most common tag
            top_tag = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[0][0]
            
            # Get snippets with this tag, excluding already recommended
            already_recommended_ids = [s.id for s in recommendations]
            tag_recommendations = []
            
            # We need to manually filter since tags are stored as comma-separated strings
            for snippet in CodeSnippet.query.filter(~CodeSnippet.id.in_(already_recommended_ids)).all():
                if snippet.tags and top_tag in [t.strip() for t in snippet.tags.split(',')]:
                    tag_recommendations.append(snippet)
                    if len(tag_recommendations) >= 2:
                        break
            
            recommendations.extend(tag_recommendations)
        
        # If we still don't have enough recommendations, add random snippets
        if len(recommendations) < limit:
            already_recommended_ids = [s.id for s in recommendations]
            remaining_needed = limit - len(recommendations)
            
            random_snippets = CodeSnippet.query.filter(
                ~CodeSnippet.id.in_(already_recommended_ids)
            ).order_by(db.func.random()).limit(remaining_needed).all()
            
            recommendations.extend(random_snippets)
        
        # Trim to the requested limit
        recommendations = recommendations[:limit]
    except Exception as e:
        logging.error(f"Error generating recommendations: {str(e)}")
    
    return recommendations


@app.route('/snippets')
def snippets_page():
    """Display the code snippets library page"""
    snippets = CodeSnippet.query.order_by(CodeSnippet.created_at.desc()).all()
    languages = db.session.query(CodeSnippet.language).distinct().all()
    languages = [lang[0] for lang in languages]
    
    # Get all tags across snippets
    all_tags = set()
    for snippet in snippets:
        if snippet.tags:
            all_tags.update([tag.strip() for tag in snippet.tags.split(',')])
    
    # Get personalized recommendations
    recommended_snippets = get_user_recommendations(limit=6)
    
    return render_template('snippets.html', 
                          active_page='snippets',
                          snippets=snippets,
                          languages=languages,
                          all_tags=sorted(all_tags),
                          recommended_snippets=recommended_snippets)

@app.route('/snippets/new', methods=['GET', 'POST'])
def new_snippet():
    """Create a new code snippet"""
    if request.method == 'POST':
        # Get form data
        title = request.form.get('title')
        description = request.form.get('description')
        code = request.form.get('code')
        language = request.form.get('language')
        tags = request.form.get('tags')
        
        # Validate required fields
        if not title or not code or not language:
            return render_template('new_snippet.html', 
                                  active_page='snippets',
                                  error="Title, code, and language are required fields.",
                                  title=title,
                                  description=description,
                                  code=code,
                                  language=language,
                                  tags=tags)
        
        # Create new snippet
        snippet = CodeSnippet(
            title=title,
            description=description,
            code=code,
            language=language,
            tags=tags
        )
        
        db.session.add(snippet)
        db.session.commit()
        
        return redirect(url_for('snippets_page'))
    
    return render_template('new_snippet.html', active_page='snippets')

@app.route('/snippets/<snippet_id>')
def view_snippet(snippet_id):
    """View a single code snippet"""
    snippet = CodeSnippet.query.get_or_404(snippet_id)
    return render_template('view_snippet.html', 
                          active_page='snippets',
                          snippet=snippet)

@app.route('/snippets/<snippet_id>/edit', methods=['GET', 'POST'])
def edit_snippet(snippet_id):
    """Edit an existing code snippet"""
    snippet = CodeSnippet.query.get_or_404(snippet_id)
    
    if request.method == 'POST':
        # Get form data
        title = request.form.get('title')
        description = request.form.get('description')
        code = request.form.get('code')
        language = request.form.get('language')
        tags = request.form.get('tags')
        
        # Validate required fields
        if not title or not code or not language:
            return render_template('edit_snippet.html', 
                                  active_page='snippets',
                                  error="Title, code, and language are required fields.",
                                  snippet=snippet)
        
        # Update snippet
        snippet.title = title
        snippet.description = description
        snippet.code = code
        snippet.language = language
        snippet.tags = tags
        snippet.updated_at = datetime.datetime.utcnow()
        
        db.session.commit()
        
        return redirect(url_for('view_snippet', snippet_id=snippet.id))
    
    return render_template('edit_snippet.html', 
                          active_page='snippets',
                          snippet=snippet)

@app.route('/snippets/<snippet_id>/delete', methods=['POST'])
def delete_snippet(snippet_id):
    """Delete a code snippet"""
    snippet = CodeSnippet.query.get_or_404(snippet_id)
    db.session.delete(snippet)
    db.session.commit()
    
    return redirect(url_for('snippets_page'))

@app.route('/api/snippets/search')
def search_snippets():
    """API endpoint to search for snippets"""
    query = request.args.get('query', '')
    language = request.args.get('language', '')
    tag = request.args.get('tag', '')
    
    # Start with all snippets
    snippets = CodeSnippet.query
    
    # Apply filters
    if query:
        snippets = snippets.filter(
            db.or_(
                CodeSnippet.title.ilike(f'%{query}%'),
                CodeSnippet.description.ilike(f'%{query}%'),
                CodeSnippet.code.ilike(f'%{query}%'),
                CodeSnippet.tags.ilike(f'%{query}%')
            )
        )
    
    if language:
        snippets = snippets.filter(CodeSnippet.language == language)
    
    if tag:
        snippets = snippets.filter(CodeSnippet.tags.ilike(f'%{tag}%'))
    
    # Execute query and convert to dict
    results = [snippet.to_dict() for snippet in snippets.all()]
    
    return jsonify({'snippets': results})

@app.route('/api/snippets/generate', methods=['POST'])
def generate_snippet():
    """API endpoint to generate code snippets using AI with context awareness"""
    data = request.json
    prompt = data.get('prompt', '')
    language = data.get('language', '')
    context = data.get('context', '')  # New parameter for context awareness
    related_snippets = data.get('related_snippets', [])  # References to other snippets
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    # If context is provided, enrich the prompt with it
    context_info = ""
    if context:
        context_info = f"""
        Consider this additional context:
        {context}
        
        Use this context to inform your solution when appropriate.
        """
    
    # If related snippets are provided, include them as references
    related_info = ""
    if related_snippets:
        # Try to fetch snippets from database
        snippet_references = []
        for snippet_id in related_snippets:
            try:
                snippet = CodeSnippet.query.get(snippet_id)
                if snippet:
                    snippet_references.append(f"Title: {snippet.title}\nLanguage: {snippet.language}\nCode:\n{snippet.code}")
            except:
                pass  # Skip if snippet doesn't exist
        
        if snippet_references:
            related_info = "Here are related code snippets to consider:\n" + "\n\n".join(snippet_references)
    
    # Identify common coding patterns from existing snippets
    popular_languages = []
    common_tags = []
    
    try:
        # Get most used languages
        language_counts = db.session.query(CodeSnippet.language, db.func.count(CodeSnippet.language))\
            .group_by(CodeSnippet.language)\
            .order_by(db.func.count(CodeSnippet.language).desc())\
            .limit(3)\
            .all()
        popular_languages = [lang for lang, count in language_counts]
        
        # Get most used tags
        all_tags = {}
        for snippet in CodeSnippet.query.all():
            if snippet.tags:
                for tag in snippet.tags.split(','):
                    tag = tag.strip()
                    all_tags[tag] = all_tags.get(tag, 0) + 1
        
        # Sort by count and get top 5
        common_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:5]
        common_tags = [tag for tag, count in common_tags]
    except:
        # If database query fails, continue without this information
        pass
    
    # Build language suggestion based on history or user preference
    lang_suggestion = ""
    if language:
        lang_suggestion = f"Use {language} as the programming language."
    elif popular_languages:
        lang_suggestion = f"Consider using one of these common languages in your codebase: {', '.join(popular_languages)}"
    
    # Construct a programming-specific prompt with all the contextual information
    programming_prompt = f"""
    Generate a code snippet based on the following request:
    
    {prompt}
    
    {context_info}
    
    {related_info}
    
    {lang_suggestion}
    
    Please provide:
    1. A descriptive title
    2. A brief explanation of what the code does
    3. The code itself with proper formatting and best practices
    4. Appropriate tags for categorization (consider these common tags if relevant: {', '.join(common_tags) if common_tags else 'algorithm, utility, function'})
    5. The programming language used
    
    Format your response as a JSON object with these fields:
    {{
      "title": "...",
      "description": "...",
      "code": "...",
      "language": "...",
      "tags": "..."
    }}
    """
    
    # Get AI response
    ai_response = get_gemini_response(programming_prompt)
    
    # Try to parse the response as JSON
    try:
        # Extract JSON object from the response
        json_str = ai_response
        
        # If the response has markdown code blocks with json
        if "```json" in json_str and "```" in json_str.split("```json", 1)[1]:
            json_str = json_str.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in json_str and "```" in json_str.split("```", 1)[1]:
            # Handle case where language isn't specified in markdown
            json_str = json_str.split("```", 1)[1].split("```", 1)[0].strip()
        
        # Parse the JSON
        snippet_data = json.loads(json_str)
        
        # Ensure all required fields are present
        required_fields = ['title', 'description', 'code', 'language']
        for field in required_fields:
            if field not in snippet_data:
                return jsonify({'error': f'Generated snippet missing {field} field', 'raw_response': ai_response}), 400
        
        # Ensure tags exists, even if empty
        if 'tags' not in snippet_data:
            snippet_data['tags'] = ''
        
        return jsonify({'snippet': snippet_data})
    except Exception as e:
        return jsonify({'error': f'Failed to parse generated snippet: {str(e)}', 'raw_response': ai_response}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Get response from Vertex
    bot_response = get_gemini_response(user_message)
    
    return jsonify({'response': bot_response})

# Facebook Messenger Webhook Routes
@app.route('/webhook', methods=['GET'])
def verify_webhook():
    """
    Facebook webhook verification endpoint.
    This endpoint is used by Facebook to verify the webhook.
    """
    # Parse the query params
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    
    # Get the verify token from environment variables
    verify_token = os.environ.get('FB_VERIFY_TOKEN', 'vertex_ai_webhook_token')
    
    logging.debug(f"Webhook verification request: mode={mode}, token={token}")
    
    # Check if a token and mode is in the query string of the request
    if mode and token:
        # Check the mode and token sent
        if mode == 'subscribe' and token == verify_token:
            logging.info("Webhook verified!")
            # Respond with the challenge token from the request
            return challenge
    
    # Responds with '403 Forbidden' if verify tokens do not match
    logging.warning("Webhook verification failed")
    return 'Forbidden', 403

@app.route('/webhook', methods=['POST'])
def webhook():
    """
    Facebook webhook endpoint to receive messages.
    This endpoint receives messages from Facebook.
    """
    # Get the request body data
    data = request.get_json()
    logging.debug(f"Received webhook data: {json.dumps(data)}")
    
    # Make sure this is a page subscription
    if data.get('object') == 'page':
        # Iterate over each entry - there may be multiple if batched
        for entry in data.get('entry', []):
            page_id = entry.get('id')
            # Get the message. entry.messaging is an array, but
            # will only ever contain one message, so we get index 0
            for messaging_event in entry.get('messaging', []):
                # Someone sent a message
                if 'message' in messaging_event:
                    # Extract the sender and message text
                    sender_id = messaging_event['sender']['id']
                    recipient_id = messaging_event['recipient']['id']
                    
                    # Extract message content if available
                    if 'text' in messaging_event.get('message', {}):
                        message_text = messaging_event['message']['text']
                        logging.info(f"Message received from {sender_id}: {message_text}")
                        
                        # Get AI response
                        ai_response = get_gemini_response(message_text)
                        
                        # Send the message
                        send_message(sender_id, ai_response)
                    else:
                        # Handle non-text messages like images, stickers etc.
                        logging.info(f"Received non-text message from {sender_id}")
                        send_message(sender_id, "I can only understand text messages at the moment.")
        
        # Return a '200 OK' response to all events
        return 'EVENT_RECEIVED', 200
    else:
        # Return a '404 Not Found' if event is not from a page subscription
        return 'Not a page subscription', 404

def send_message(recipient_id, message_text):
    """
    Send message to a specific recipient via Facebook Messenger.
    
    Args:
        recipient_id (str): Facebook user ID to send message to
        message_text (str): Message text to send
    """
    # Get the page access token from env vars
    page_access_token = os.environ.get('FB_PAGE_ACCESS_TOKEN')
    
    if not page_access_token:
        logging.error("FB_PAGE_ACCESS_TOKEN not found in environment variables")
        return False
    
    logging.debug(f"Sending message to {recipient_id}: {message_text}")
    
    # Construct the message payload
    params = {
        "access_token": page_access_token
    }
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message_text
        }
    }
    
    # Send the message
    try:
        response = requests.post(
            "https://graph.facebook.com/v18.0/me/messages",
            params=params,
            headers=headers,
            json=data
        )
        
        # Check for successful response
        if response.status_code == 200:
            logging.info(f"Message sent successfully to {recipient_id}")
            return True
        else:
            logging.error(f"Failed to send message: {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"Exception when sending message: {str(e)}")
        return False

if __name__ == '__main__':
    # Run the app on port 5000 and bind to all interfaces
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
