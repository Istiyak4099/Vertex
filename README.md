# Gemini AI Chatbot

A Flask-based web application that integrates with Google's Gemini API to create an interactive chatbot with a clean, responsive UI.

## Features

- Interactive chat interface with real-time responses
- Integration with Google's Gemini AI models (automatically selects the best available model)
- Responsive design using Bootstrap with Replit dark theme
- Support for code blocks and markdown in responses
- Configured for easy deployment on Render.com

## Live Demo

Visit the deployed application at [your-app-url-here] once deployed.

## Project Structure

```
.
├── app.py               # Main Flask application
├── main.py              # Entry point for Gunicorn
├── templates/           # HTML templates
│   └── index.html       # Chat interface
├── requirements.txt     # Project dependencies
├── render.yaml          # Render deployment configuration
└── .gitignore           # Git ignore file
```

## Requirements

- Python 3.7+
- Flask
- Google Generative AI Python SDK
- Gunicorn (for production deployment)

## Environment Variables

The application requires the following environment variables:

- `GOOGLE_API_KEY`: Your Google Gemini API key (get one at https://makersuite.google.com/app/apikey)
- `SESSION_SECRET`: Secret key for Flask session (auto-generated on Render)

## Setup and Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up the environment variables:
   ```
   export GOOGLE_API_KEY=your_key_here
   export SESSION_SECRET=your_secret_here
   ```
4. Run the application:
   ```
   python app.py
   ```
   Or in production with Gunicorn:
   ```
   gunicorn app:app -b 0.0.0.0:5000
   ```

## Deployment on Render.com

This project includes a `render.yaml` file for easy deployment on Render.com:

1. Push this repository to GitHub
2. Create a new Web Service on Render.com
3. Connect your GitHub repository
4. Render will automatically detect the configuration
5. Add your `GOOGLE_API_KEY` in the environment variables section
6. Deploy the application

## Usage

1. Visit the application in your browser
2. Type a message in the input field and hit send
3. Receive AI-generated responses from Google's Gemini model
4. The chat supports code blocks, formatting, and other markdown features

## Screenshots

(Add screenshots of your application here)

## License

MIT

## Author

[Your Name]