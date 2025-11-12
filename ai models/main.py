from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
import base64
from io import BytesIO
from dotenv import load_dotenv
import requests
import uuid
import json 
import re
import time
from sarvasva_core import SarvasvaChatbot, TTS_CONFIGS 
from video_processor import start_async_video_pipeline 
from pdf2image import convert_from_bytes 
import pytesseract
from PIL import Image
load_dotenv()
DEFAULT_LANG = "en-IN" 
app = Flask(__name__, static_folder='static', template_folder="templates")
CORS(app)
logging.basicConfig(level=logging.INFO)
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
STT_API_URL = "https://api.sarvam.ai/speech-to-text"
if not SARVAM_API_KEY or not GROQ_API_KEY:
    logging.error("FATAL: API keys not loaded. Check .env file.")
else:
    # Log API key status without exposing the key
    logging.info(f"Sarvam API Key: {'✓ Loaded' if SARVAM_API_KEY else '✗ Missing'} (length: {len(SARVAM_API_KEY) if SARVAM_API_KEY else 0})")
    logging.info(f"Groq API Key: {'✓ Loaded' if GROQ_API_KEY else '✗ Missing'} (length: {len(GROQ_API_KEY) if GROQ_API_KEY else 0})")
chatbots = {} 
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'webm'}
def get_chatbot(session_id):
    """Retrieves or creates a SarvasvaChatbot instance for a session."""
    if session_id not in chatbots:
        try:
            new_bot = SarvasvaChatbot(
                groq_api_key=GROQ_API_KEY, 
                sarvam_api_key=SARVAM_API_KEY,
                language_code=DEFAULT_LANG 
            )
            chatbots[session_id] = new_bot
            logging.info(f"New chatbot instance created for session {session_id}")
        except ValueError as e:
            logging.error(f"Failed to initialize chatbot: {e}")
            return None
    return chatbots[session_id]
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/') 
def home():
    """ Serve the frontend HTML file """
    return render_template("index.html")
@app.route('/set-language', methods=['POST'])
def set_language():
    """Sets the language code for the specified session."""
    data = request.json
    new_lang = data.get("language_code", DEFAULT_LANG).strip()
    session_id = data.get("session_id", str(uuid.uuid4()))
    if new_lang not in TTS_CONFIGS: 
        return jsonify({"error": "Invalid or unsupported language code"}), 400
    chatbot = get_chatbot(session_id)
    chatbot.set_language(new_lang)
    logging.info(f"Session {session_id} language set to: {new_lang}")
    return jsonify({"message": f"Language changed to {new_lang}"}), 200

@app.route('/supported-languages', methods=['GET'])
def get_supported_languages():
    """Returns a list of all supported languages with their configurations."""
    languages = []
    language_names = {
        'en-IN': 'English (India)',
        'hi-IN': 'Hindi',
        'ta-IN': 'Tamil',
        'bn-IN': 'Bengali',
        'gu-IN': 'Gujarati',
        'kn-IN': 'Kannada',
        'ml-IN': 'Malayalam',
        'mr-IN': 'Marathi',
        'od-IN': 'Odia',
        'pa-IN': 'Punjabi',
        'te-IN': 'Telugu',
    }
    for lang_code, config in TTS_CONFIGS.items():
        languages.append({
            "code": lang_code,
            "name": language_names.get(lang_code, lang_code),
            "speaker": config.get("speaker", "default"),
            "model": config.get("model", "bulbul:v2")
        })
    return jsonify({"languages": languages, "default": DEFAULT_LANG}), 200
@app.route('/chat', methods=['POST'])
def chat():
    """
    Primary chat endpoint. Handles text or voice input and returns text and audio output.
    Supports multilingual conversations with automatic TTS for all responses.
    (Used for Feature 2 Chat/Doubts, Feature 5 P2P is client-side.)
    """
    data = request.json
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", str(uuid.uuid4()))
    reset = data.get("reset", False)
    is_voice_chat = data.get("is_voice_chat", False)
    language_code = data.get("language_code", None)  # Optional: override language for this request
    
    chatbot = get_chatbot(session_id)
    if chatbot is None:
        return jsonify({"error": "Chatbot service is unavailable. API keys might be missing."}), 503
    
    # Set language if provided
    if language_code and language_code in TTS_CONFIGS:
        chatbot.set_language(language_code)
        logging.info(f"Language set to {language_code} for session {session_id}")
    
    if reset:
        chatbot.reset_session()
    
    try:
        # Always generate TTS audio for multilingual support
        # Auto-detect language from input text (enabled by default)
        response_data = chatbot.generate_response(
            user_message_vernacular=user_message, 
            is_voice_chat=is_voice_chat,
            always_tts=True,  # Always generate TTS audio
            auto_detect_language=True  # Auto-detect and switch language from input
        )
        return jsonify({
            "response": response_data['text'],
            "audio_response": response_data['audio_base64'],
            "session_id": session_id,
            "memory_turns": response_data['history_length'],
            "language_code": response_data.get('language_code', chatbot.language_code),
            "has_audio": response_data['audio_base64'] is not None
        })
    except Exception as e:
        logging.error(f"Unexpected error in /chat for session {session_id}: {str(e)}")
        if "Authentication" in str(e) or "API key" in str(e):
             return jsonify({"error": "LLM Authentication Error. Check GROQ_API_KEY."}), 503
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    """Converts uploaded audio file (Voice Input) to text using Sarvam STT."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    audio_file = request.files['audio']
    if audio_file.filename == '' or not allowed_file(audio_file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    current_lang = request.form.get('language_code', DEFAULT_LANG)
    file_path = None
    
    try:
        # Check API key
        if not SARVAM_API_KEY:
            return jsonify({'error': 'Sarvam API key not configured. Please set SARVAM_API_KEY in .env file.'}), 500
        
        # Use BytesIO to avoid file system issues on Windows
        audio_bytes = audio_file.read()
        audio_file.seek(0)  # Reset for potential reuse
        
        # Generate unique filename to avoid conflicts
        unique_filename = f"stt_{int(time.time() * 1000)}_{secure_filename(audio_file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        audio_file.save(file_path)
        
        # Prepare request - try different header formats
        # Sarvam API might use different header names, try both
        headers = {
            'api-subscription-key': SARVAM_API_KEY,
            'Authorization': f'Bearer {SARVAM_API_KEY}',  # Alternative format
        }
        data = {'model': 'saarika:v2', 'language_code': current_lang}
        
        # Log request details (without exposing full API key)
        logging.info(f"STT Request - Language: {current_lang}, File: {unique_filename}, API Key present: {bool(SARVAM_API_KEY)}")
        
        # Read file and send request
        with open(file_path, 'rb') as f:
            files = {'file': (unique_filename, f, 'audio/webm')}  # Changed to webm as that's what frontend sends
            try:
                # Try with api-subscription-key first
                response = requests.post(
                    STT_API_URL, 
                    headers={'api-subscription-key': SARVAM_API_KEY}, 
                    data=data, 
                    files=files, 
                    timeout=30
                )
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    # Log response details for debugging
                    try:
                        error_detail = e.response.json()
                        logging.error(f"STT API 403 Forbidden - Response: {error_detail}")
                    except:
                        logging.error(f"STT API 403 Forbidden - Status: {e.response.status_code}, Headers: {dict(e.response.headers)}")
                    
                    # Check if API key is actually set
                    if not SARVAM_API_KEY:
                        error_msg = 'SARVAM_API_KEY is not set in .env file. Please add your API key.'
                    elif len(SARVAM_API_KEY) < 10:
                        error_msg = 'SARVAM_API_KEY appears to be invalid (too short). Please check your .env file.'
                    else:
                        error_msg = 'API authentication failed. Please verify your SARVAM_API_KEY is correct and has STT permissions.'
                    
                    return jsonify({'error': error_msg}), 403
                raise
            except requests.exceptions.ConnectionError as e:
                logging.error(f"STT API connection failed: {str(e)}")
                return jsonify({
                    'error': 'Failed to connect to speech-to-text service. Please check your internet connection and try again.'
                }), 503
        
        result = response.json()
        transcription_text = result.get('transcript', '')
        detected_lang = result.get('language_code', current_lang)  # STT API may return detected language
        
        if not transcription_text:
            return jsonify({'error': 'No clear speech detected or API failed to transcribe.'}), 500
        
        # Auto-detect language from transcribed text if STT didn't return it
        if detected_lang == current_lang and transcription_text:
            # Try to detect language from the transcribed text
            session_id = request.form.get('session_id', None)
            if session_id:
                chatbot = get_chatbot(session_id)
                if chatbot:
                    detected_lang = chatbot.detect_language(transcription_text)
                    if detected_lang and detected_lang in TTS_CONFIGS:
                        chatbot.set_language(detected_lang)
                        logging.info(f"Auto-detected and switched to language: {detected_lang} from voice input")
                        current_lang = detected_lang
        
        return jsonify({
            'transcription': transcription_text, 
            'language_code': current_lang,
            'detected_language': detected_lang
        })
        
    except requests.exceptions.RequestException as e:
        logging.error(f"STT API request failed: {str(e)}")
        error_msg = str(e)
        if '403' in error_msg or 'Forbidden' in error_msg:
            return jsonify({
                'error': 'API authentication failed. Please check your SARVAM_API_KEY in .env file.'
            }), 403
        elif 'Connection' in error_msg or 'timeout' in error_msg.lower():
            return jsonify({
                'error': 'Failed to connect to speech-to-text service. Please check your internet connection.'
            }), 503
        return jsonify({'error': f'API request failed: {error_msg}'}), 500
    except Exception as e:
        logging.error(f"Unexpected STT error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    finally:
        # Safely delete file with retry mechanism for Windows
        if file_path and os.path.exists(file_path):
            try:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        os.remove(file_path)
                        break
                    except PermissionError:
                        if attempt < max_retries - 1:
                            time.sleep(0.1)  # Wait 100ms before retry
                        else:
                            logging.warning(f"Could not delete temporary file: {file_path}")
            except Exception as e:
                logging.warning(f"Error deleting temporary file {file_path}: {str(e)}")
@app.route('/api/check-keys', methods=['GET'])
def check_api_keys():
    """Diagnostic endpoint to check API key configuration."""
    return jsonify({
        'sarvam_api_key': {
            'loaded': bool(SARVAM_API_KEY),
            'length': len(SARVAM_API_KEY) if SARVAM_API_KEY else 0,
            'preview': f"{SARVAM_API_KEY[:8]}..." if SARVAM_API_KEY and len(SARVAM_API_KEY) > 8 else "Not set"
        },
        'groq_api_key': {
            'loaded': bool(GROQ_API_KEY),
            'length': len(GROQ_API_KEY) if GROQ_API_KEY else 0,
            'preview': f"{GROQ_API_KEY[:8]}..." if GROQ_API_KEY and len(GROQ_API_KEY) > 8 else "Not set"
        },
        'stt_endpoint': STT_API_URL
    }), 200

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech_endpoint():
    """Converts text input directly to audio (for initial greetings, etc.)."""
    data = request.json
    text_input = data.get("text", "").strip()
    target_lang = data.get("target_language_code", DEFAULT_LANG)
    if not text_input:
        return jsonify({"error": "Text is required"}), 400
    chatbot = get_chatbot("temp_tts_session") 
    base64_audio_string = chatbot._tts(text_input, target_lang)
    if base64_audio_string:
        audio_bytes = base64.b64decode(base64_audio_string)
        audio_data_combined = BytesIO(audio_bytes)
        audio_data_combined.seek(0)
        return send_file(audio_data_combined, mimetype="audio/mpeg")
    else:
        return jsonify({"error": "Failed to generate audio"}), 500
@app.route('/api/generate-quiz', methods=['POST'])
def generate_quiz_endpoint():
    """Endpoint for Feature 6: AI Quiz Generation."""
    data = request.json
    topic = data.get("topic")
    session_id = data.get("session_id", "quiz_session")
    if not topic:
        return jsonify({"error": "Topic is required for quiz generation."}), 400
    chatbot = get_chatbot(session_id)
    if chatbot is None: return jsonify({"error": "LLM service unavailable."}), 503
    try:
        quiz_json_string = chatbot.generate_quiz(topic)
        return jsonify({"quiz_data": json.loads(quiz_json_string)})
    except json.JSONDecodeError:
         return jsonify({"error": "LLM returned invalid JSON structure. Check console."}), 500
    except Exception as e:
        logging.error(f"Quiz Generation API Error: {e}")
        return jsonify({"error": f"Quiz generation failed: {str(e)}"}), 500
@app.route('/api/summarize', methods=['POST'])
def summarize_resource_endpoint():
    """Endpoint for Feature 4: Resource Summarizer."""
    data = request.json
    resource_text = data.get("resource_text")
    session_id = data.get("session_id", "summary_session")
    target_lang = data.get("language_code", DEFAULT_LANG)
    if not resource_text:
        return jsonify({"error": "Resource text is required for summarization."}), 400
    chatbot = get_chatbot(session_id)
    if chatbot is None: return jsonify({"error": "LLM service unavailable."}), 503
    try:
        summary_text = chatbot.summarize_resource(resource_text, target_lang)
        return jsonify({"summary": summary_text})
    except Exception as e:
        logging.error(f"Summarization API Error: {e}")
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500
@app.route('/api/explain-timestamp', methods=['POST'])
def explain_timestamp_endpoint():
    """
    Endpoint for Feature 2: Provides real-time explanation based on video context/timestamp.
    """
    data = request.json
    session_id = data.get("session_id", "timestamp_session")
    course_topic = data.get("course_topic")
    video_transcript = data.get("transcript_segment")
    timestamp = data.get("timestamp")
    user_doubt = data.get("user_doubt")
    target_lang = data.get("language_code", DEFAULT_LANG)
    if not all([course_topic, video_transcript, timestamp, user_doubt]):
        return jsonify({"error": "Missing required fields: topic, transcript, timestamp, or doubt."}), 400
    chatbot = get_chatbot(session_id)
    if chatbot is None: return jsonify({"error": "LLM service unavailable."}), 503
    try:
        explanation_result = chatbot.explain_timestamp_doubt(
            course_topic, 
            video_transcript, 
            timestamp, 
            user_doubt, 
            target_lang
        )
        return jsonify(explanation_result)
    except Exception as e:
        logging.error(f"Timestamp Explanation Endpoint Error: {e}")
        return jsonify({"error": f"Failed to provide explanation: {str(e)}"}), 500
@app.route('/api/upload-course', methods=['POST'])
def upload_course():
    """
    Endpoint for Feature 1: Starts the asynchronous video translation pipeline (MOCK).
    This demonstrates the instructor-side functionality.
    """
    data = request.json
    course_title = data.get("course_title")
    target_lang = data.get("language_code", DEFAULT_LANG)
    instructor_id = data.get("user_id", "instructor_mock")
    if not course_title:
        return jsonify({"error": "Course title is required."}), 400
    mock_video_path = f"/storage/{instructor_id}/{course_title.replace(' ', '_')}.mp4"
    try:
        job_status = start_async_video_pipeline(
            video_path=mock_video_path,
            course_id=str(uuid.uuid4()),
            target_lang=target_lang
        )
        return jsonify({
            "status": "Success",
            "message": f"Course '{course_title}' submitted for AI processing.",
            "pipeline_details": job_status
        })
    except Exception as e:
        logging.error(f"Course Upload/Pipeline Initiation Error: {e}")
        return jsonify({"error": "Failed to start AI pipeline."}), 500
@app.route('/api/peer-chat', methods=['POST'])
def peer_chat():
    """Endpoint for Feature 5: Peer-to-Peer Message Sending (Placeholder)."""
    logging.warning("Peer chat backend is currently a placeholder.")
    return jsonify({"status": "Success (Mocked for Demo)"})
def perform_ocr(file_bytes, file_type):
    logging.warning("Attempted local OCR which is not supported on Vercel. Failing gracefully.")
    raise Exception("Document analysis is currently disabled in the cloud.")
@app.route('/read-document', methods=['POST'])
def read_document():
    """Route for Feature 4 Upload/Analysis (Mocked)."""
    try:
        perform_ocr(None, None) 
        return jsonify({"status": "processing"})
    except Exception as e:
        return jsonify({"error": f"Failed to process document: {str(e)}"}), 500
if __name__ == '__main__':
    logging.info("Starting SARVASVA AI Service...")
    AI_PORT = int(os.getenv('AI_PORT', 5001))
    app.run(host='127.0.0.1', port=AI_PORT, debug=True)