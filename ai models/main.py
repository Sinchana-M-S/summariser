from flask import Flask, request, jsonify, send_file
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
from sarvasva_core import SarvasvaChatbot, TTS_CONFIGS, GROQ_MODEL_CHAT 
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
TRANSLATE_API_URL = "https://api.sarvam.ai/translate"
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
    """ API health check endpoint """
    return jsonify({
        "status": "running",
        "service": "Sarvasva AI API",
        "version": "1.0",
        "endpoints": {
            "chat": "/chat",
            "speech-to-text": "/speech-to-text",
            "text-to-speech": "/text-to-speech",
            "translate": "/translate",
            "read-document": "/read-document",
            "set-language": "/set-language"
        }
    })
@app.route('/set-language', methods=['POST'])
def set_language():
    """Set the default language for the application and reset conversation session - matches SARVASVA"""
    data = request.json
    new_lang = data.get("language_code", "").strip()
    session_id = data.get("session_id", "default")

    if not new_lang:
        return jsonify({"error": "Language code is required"}), 400

    # Validate language code
    valid_languages = ["en-IN", "hi-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"]
    if new_lang not in valid_languages:
        return jsonify({"error": f"Invalid language code. Must be one of: {', '.join(valid_languages)}"}), 400

    logging.info(f"Language set to: {new_lang}, resetting session: {session_id}")
    
    # Reset the conversation session for fresh start
    if session_id in chatbots:
        del chatbots[session_id]
    
    # Create new chatbot with the new language
    chatbot = get_chatbot(session_id)
    chatbot.set_language(new_lang)
    
    return jsonify({
        "message": f"Language changed to {new_lang}",
        "language_code": new_lang,
        "session_reset": True
    }), 200

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
    is_initial_greeting = data.get("message") == "initial_greeting" or data.get("isGreeting", False)
    
    chatbot = get_chatbot(session_id)
    if chatbot is None:
        return jsonify({"error": "Chatbot service is unavailable. API keys might be missing."}), 503
    
    # Priority: Use provided language_code (from voice STT), otherwise auto-detect from text input
    detected_language = None
    if language_code and language_code in TTS_CONFIGS:
        detected_language = language_code
        chatbot.set_language(language_code)
        logging.info(f"Language set to {language_code} for session {session_id} (from request)")
    else:
        # Auto-detect language from input text if not provided (skip for initial greeting)
        if not is_initial_greeting:
            detected_language = chatbot.detect_language(user_message)
            if detected_language and detected_language in TTS_CONFIGS:
                chatbot.set_language(detected_language)
                logging.info(f"Language auto-detected and set to {detected_language} for session {session_id} from text: '{user_message[:50]}...'")
            else:
                # Fallback to default if detection fails
                detected_language = chatbot.language_code
                logging.info(f"Language detection failed, using current/default: {detected_language}")
        else:
            detected_language = chatbot.language_code
    
    if reset:
        chatbot.reset_session()
    
    # Validate message (allow initial_greeting even if empty)
    if not user_message and not is_initial_greeting:
        return jsonify({"error": "Message cannot be empty"}), 400
    
    # Ensure detected_language is always set
    if detected_language is None:
        detected_language = chatbot.language_code
        logging.warning(f"Detected language was None, using default: {detected_language}")
    
    try:
        # Always generate TTS audio for multilingual support
        # Language is already set above, so generate_response will use it
        logging.info(f"Generating response for session {session_id}, message: '{user_message[:50]}...', language: {detected_language}")
        response_data = chatbot.generate_response(
            user_message_vernacular=user_message if not is_initial_greeting else "", 
            is_voice_chat=is_voice_chat,
            always_tts=True,  # Always generate TTS audio
            auto_detect_language=False,  # Language already set above, don't detect again
            is_initial_greeting=is_initial_greeting
        )
        
        # Ensure the response language code matches what we detected/set
        response_language = response_data.get('language_code', chatbot.language_code)
        if response_language != detected_language:
            logging.warning(f"Response language ({response_language}) doesn't match detected language ({detected_language}), using detected language")
            response_data['language_code'] = detected_language
        
        # Validate response data
        if not response_data or 'text' not in response_data:
            logging.error(f"Invalid response_data from generate_response: {response_data}")
            return jsonify({"error": "Invalid response from AI service"}), 500
        
        logging.info(f"✅ Successfully generated response for session {session_id}, text length: {len(response_data.get('text', ''))}, has_audio: {response_data.get('audio_base64') is not None}")
        return jsonify({
            "response": response_data['text'],
            "audio_response": response_data.get('audio_base64'),
            "session_id": session_id,
            "memory_turns": response_data.get('history_length', 0),
            "language_code": response_data.get('language_code', chatbot.language_code),
            "has_audio": response_data.get('audio_base64') is not None,
            "questions_asked": chatbot.question_count,
            "assessment_provided": chatbot.assessment_provided
        })
    except ValueError as e:
        logging.error(f"ValueError in /chat for session {session_id}: {str(e)}")
        if "API keys" in str(e):
            return jsonify({"error": "API keys are missing. Please check your .env file."}), 503
        return jsonify({"error": "Configuration error", "details": str(e)}), 500
    except Exception as e:
        logging.error(f"Unexpected error in /chat for session {session_id}: {str(e)}", exc_info=True)
        if "Authentication" in str(e) or "API key" in str(e) or "401" in str(e) or "403" in str(e):
             return jsonify({"error": "LLM Authentication Error. Check GROQ_API_KEY and SARVAM_API_KEY."}), 503
        if "Connection" in str(e) or "timeout" in str(e).lower():
            return jsonify({"error": "AI service connection error. Please check your internet connection and API service status."}), 503
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    """Convert Speech to Text - matches SARVASVA implementation exactly"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(audio_file.filename):
        return jsonify({'error': 'Invalid file format'}), 400

    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_path_initialized = False

    try:
        audio_file.save(file_path)
        file_path_initialized = True

        if os.stat(file_path).st_size == 0:
            os.remove(file_path)
            return jsonify({'error': 'Uploaded file is empty'}), 400

        current_lang = request.form.get('language_code', DEFAULT_LANG)
        logging.info(f"Using language for STT: {current_lang}")

        # Call Speech-to-Text API
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, 'audio/wav')}
            data = {
                'model': 'saarika:v2',
                'language_code': current_lang,
                'with_timestamps': 'false',
                'with_diarization': 'false',
                'num_speakers': '1'
            }
            headers = {'api-subscription-key': SARVAM_API_KEY}
            response = requests.post('https://api.sarvam.ai/speech-to-text', headers=headers, data=data, files=files)
            response.raise_for_status()

            result = response.json()
            logging.info(f"Speech-to-text response: {result}")

        if 'transcript' not in result:
            return jsonify({'error': 'No transcript found in response'}), 500

        transcription_text = result.get('transcript', '')
        detected_language = result.get('language_code', current_lang)

        # Ensure we have a transcription
        if not transcription_text:
            logging.warning("STT API returned empty transcript")
            return jsonify({'error': 'No transcription found in response', 'transcription': '', 'language_code': current_lang}), 500

        response_data = {
            'transcription': transcription_text,
            'language_code': detected_language,
            'detected_language': detected_language  # Add for frontend compatibility
        }

        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        logging.error(f"Speech-to-text API request failed: {str(e)}")
        return jsonify({'error': f'API request failed: {str(e)}'}), 500

    except Exception as e:
        logging.error(f"Unexpected error in STT: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

    finally:
        if file_path_initialized and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logging.warning(f"Could not delete temporary file {file_path}: {str(e)}")
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

@app.route('/diagnostics', methods=['GET'])
def diagnostics():
    """Comprehensive diagnostic endpoint to test chatbot functionality."""
    diagnostics_result = {
        'api_keys': {
            'sarvam': {
                'loaded': bool(SARVAM_API_KEY),
                'length': len(SARVAM_API_KEY) if SARVAM_API_KEY else 0
            },
            'groq': {
                'loaded': bool(GROQ_API_KEY),
                'length': len(GROQ_API_KEY) if GROQ_API_KEY else 0
            }
        },
        'chatbot_initialization': None,
        'groq_test': None,
        'errors': []
    }
    
    # Test chatbot initialization
    try:
        test_bot = get_chatbot("diagnostic_test")
        if test_bot:
            diagnostics_result['chatbot_initialization'] = {
                'status': 'success',
                'language': test_bot.language_code,
                'model_chat': test_bot.model_name_chat
            }
        else:
            diagnostics_result['chatbot_initialization'] = {'status': 'failed', 'reason': 'get_chatbot returned None'}
            diagnostics_result['errors'].append('Chatbot initialization failed')
    except Exception as e:
        diagnostics_result['chatbot_initialization'] = {'status': 'error', 'error': str(e)}
        diagnostics_result['errors'].append(f'Chatbot initialization error: {str(e)}')
    
    # Test Groq API connection
    if GROQ_API_KEY:
        try:
            from groq import Groq
            test_groq = Groq(api_key=GROQ_API_KEY)
            test_completion = test_groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Say 'test'"}],
                max_tokens=10
            )
            diagnostics_result['groq_test'] = {
                'status': 'success',
                'response': test_completion.choices[0].message.content[:50]
            }
        except Exception as e:
            diagnostics_result['groq_test'] = {
                'status': 'error',
                'error': str(e)
            }
            diagnostics_result['errors'].append(f'Groq API test error: {str(e)}')
    else:
        diagnostics_result['groq_test'] = {'status': 'skipped', 'reason': 'GROQ_API_KEY not set'}
    
    return jsonify(diagnostics_result), 200

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    """Convert Text to Speech using Sarvam AI - matches SARVASVA implementation exactly"""
    try:
        data = request.json
        text_list = data.get("inputs", [])
        if not text_list or not isinstance(text_list, list) or not text_list[0].strip():
            return jsonify({"error": "Text is required"}), 400

        text = text_list[0]
        should_summarize = data.get("summarize", False)
        
        currLang = data.get("target_language_code", DEFAULT_LANG)
        source_lang = data.get("source_language_code", DEFAULT_LANG)

        # Count words in text
        word_count = len(text.split())
        logging.info(f"TTS request: word_count={word_count}, should_summarize={should_summarize}")
        
        # Summarize long text if more than 100 words (automatic for TTS to keep audio reasonable)
        if word_count > 100:
            try:
                logging.info(f"Starting summarization for {word_count} words")
                summarize_prompt = f"""You are summarizing an educational response for text-to-speech. The original text has {word_count} words.

CRITICAL REQUIREMENTS:
1. Summarize to approximately 50 words (40-60 words acceptable)
2. Cover ALL important points, concepts, and key information from the original text
3. Do NOT omit any significant details - include all main topics, examples, and conclusions
4. Maintain the same language and tone as the original
5. Make it natural for speech - use complete sentences
6. Ensure the summary is comprehensive despite being short

Original text to summarize:

{text}

Provide a comprehensive summary of approximately 50 words that includes ALL important information, key points, main concepts, and significant details from the original text:"""
                
                chatbot = get_chatbot("temp_tts_session")
                if chatbot is None:
                    raise Exception("Failed to initialize chatbot for summarization")
                summary_completion = chatbot.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": summarize_prompt}],
                    model=GROQ_MODEL_CHAT,
                    temperature=0.3,
                    max_tokens=200
                )
                original_word_count = word_count
                summarized_text = summary_completion.choices[0].message.content.strip()
                new_word_count = len(summarized_text.split())
                
                # Validate summary is reasonable length
                if new_word_count > 0 and new_word_count <= 80:
                    text = summarized_text
                    logging.info(f"TTS text successfully summarized from {original_word_count} words to {new_word_count} words")
                else:
                    logging.warning(f"Summary length unexpected ({new_word_count} words), using original text")
                    # Fall back to truncation if summary is too long
                    if word_count > 100:
                        words = text.split()
                        text = ' '.join(words[:100]) + "..."
            except Exception as e:
                logging.error(f"TTS summarization failed: {str(e)}")
                # Continue with original text if summarization fails
                # But truncate to reasonable length for TTS (approximately 100 words)
                if word_count > 100:
                    words = text.split()
                    text = ' '.join(words[:100]) + "..."
                    logging.info(f"Fallback: truncated text to 100 words")

        # Use TTS_CONFIGS (already imported at top)
        config = TTS_CONFIGS.get(currLang, TTS_CONFIGS['en-IN'])
        model = config["model"]
        chunk_size = config["chunk_size"]
        silence_bytes = config["silence_bytes"]
        speaker = config["speaker"]

        # 1. Translate text if source and target languages differ
        if source_lang != currLang:
            translate_payload = {
                "input": text,
                "source_language_code": source_lang,
                "target_language_code": currLang,
                "speaker_gender": "Female",
                "mode": "formal",
                "model": "bulbul:v1"
            }
            translate_headers = {
                "Content-Type": "application/json",
                "api-subscription-key": SARVAM_API_KEY
            }
            try:
                translate_response = requests.post(TRANSLATE_API_URL, json=translate_payload, headers=translate_headers)
                if translate_response.status_code == 200:
                    translate_result = translate_response.json()
                    text = translate_result.get("translated_text", text)
                else:
                    logging.warning(f"Translation failed for TTS with status {translate_response.status_code}")
            except Exception as e:
                logging.error(f"Translation error in TTS: {str(e)}")
                # Continue with original text if translation fails

        # 2. Process text in chunks for TTS
        audio_data_combined = BytesIO()
        silence_chunk = b"\x00" * silence_bytes
        text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        for chunk in text_chunks:
            if not chunk.strip():
                continue

            request_body = {
                "inputs": [chunk],
                "target_language_code": currLang,
                "speaker": speaker,
                "pitch": 0,
                "pace": 1.0,
                "loudness": 1.0,
                "speech_sample_rate": 22050,
                "enable_preprocessing": True,
                "model": model
            }
            if currLang == "en-IN":
                request_body["eng_interpolation_wt"] = 123

            headers = {
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json"
            }

            response = requests.post("https://api.sarvam.ai/text-to-speech", headers=headers, json=request_body)
            if response.status_code != 200:
                logging.error(f"TTS API error for chunk: {response.text}")
                continue

            result = response.json()
            if "audios" in result and result["audios"]:
                audio_data_combined.write(base64.b64decode(result["audios"][0]))
                audio_data_combined.write(silence_chunk)

        if audio_data_combined.getbuffer().nbytes <= silence_bytes:
            return jsonify({"error": "Failed to generate audio"}), 500

        audio_data_combined.seek(0)
        return send_file(audio_data_combined, mimetype="audio/mpeg")

    except requests.exceptions.RequestException as e:
        logging.error(f"TTS API request failed: {str(e)}")
        return jsonify({"error": "API request failed", "details": str(e)}), 500

    except Exception as e:
        logging.error(f"Unexpected error in TTS: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

def perform_translation(input_text, source_lang, target_lang, speaker_gender, mode, output_script, numerals_format):
    """Perform translation request to Sarvam AI API (Helper for /translate)"""
    try:
        payload = {
            "input": input_text,
            "source_language_code": source_lang,
            "target_language_code": target_lang,
            "speaker_gender": speaker_gender,
            "mode": mode,
            "model": "mayura:v1",
            "enable_preprocessing": False,
            "output_script": output_script,
            "numerals_format": numerals_format
        }

        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": SARVAM_API_KEY
        }

        response = requests.post(TRANSLATE_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        response_data = response.json()
        
        if "translated_text" in response_data:
            result = {
                "translated_text": response_data["translated_text"],
                "request_id": response_data.get("request_id", "unknown"),
                "source_language_code": response_data.get("source_language_code", source_lang)
            }
            return result

        # Handle error response - return original text
        error_msg = response_data.get("error", {}).get("message", "Translation failed")
        logging.warning(f"Translation API error: {error_msg}, returning original text")
        return {"error": error_msg, "translated_text": input_text, "details": response_data}

    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg, "translated_text": input_text}
    except Exception as e:
        error_msg = f"Translation error: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg, "translated_text": input_text}

def translate_long_text(input_text, source_lang, target_lang, speaker_gender, mode, output_script, numerals_format):
    """Handle translation of texts longer than 1000 characters by splitting into chunks (Helper for /translate)"""
    sentences = re.split(r'(?<=[.!?])\s+', input_text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 950:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    translated_chunks = []

    for chunk in chunks:
        response = perform_translation(
            chunk, 
            source_lang, 
            target_lang, 
            speaker_gender, 
            mode, 
            output_script, 
            numerals_format
        )
        
        if "translated_text" in response:
            translated_chunks.append(response["translated_text"])
        else:
            return response
    
    full_translation = " ".join(translated_chunks)
    
    result_dict = {
        "translated_text": full_translation,
        "chunked_translation": True,
        "chunks_count": len(chunks)
    }
    
    return result_dict

@app.route('/translate', methods=['POST'])
def translate_text():
    """API to translate text using Sarvam AI - matches SARVASVA implementation exactly"""
    try:
        data = request.json
        input_text = data.get("input")
        source_lang = data.get("source_language_code", "").strip()
        target_lang = data.get("target_language_code", "").strip()
        speaker_gender = data.get("speaker_gender", "Female")
        mode = data.get("mode", "formal")
        output_script = data.get("output_script", "fully-native")
        numerals_format = data.get("numerals_format", "international")

        if not input_text or not input_text.strip():
            return jsonify({"error": "Input text is required"}), 400

        # Validate language codes
        valid_languages = ["en-IN", "hi-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"]
        if source_lang and source_lang not in valid_languages:
            return jsonify({"error": f"Invalid source language code. Must be one of: {', '.join(valid_languages)}"}), 400
        if target_lang and target_lang not in valid_languages:
            return jsonify({"error": f"Invalid target language code. Must be one of: {', '.join(valid_languages)}"}), 400

        # If source and target are same, return original text
        if source_lang == target_lang:
            return jsonify({"translated_text": input_text}), 200

        if len(input_text) > 1000:
            result = translate_long_text(input_text, source_lang, target_lang, speaker_gender, mode, output_script, numerals_format)
            return jsonify(result)
        
        result = perform_translation(input_text, source_lang, target_lang, speaker_gender, mode, output_script, numerals_format)
        if "error" in result:
            return jsonify({
                "error": result["error"],
                "translated_text": result["translated_text"],
                "request_id": result.get("request_id", "unknown"),
                "details": result.get("details", {})
            }), 400
        return jsonify(result)

    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

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
    """Perform OCR using Tesseract, handling PDF to image conversion via Poppler."""
    raw_text = ""
    
    if 'pdf' in file_type.lower():
        try:
            # Try to get poppler path from environment or use default
            poppler_path_env = os.getenv('POPPLER_PATH') or os.getenv('POPPLER_BIN')
            poppler_path_default = r'C:\Program Files\poppler\Library\bin' if os.name == 'nt' else None
            
            poppler_path = poppler_path_env or poppler_path_default

            # If the configured path exists, pass it explicitly
            if os.name == 'nt' and poppler_path and os.path.isdir(poppler_path):
                images = convert_from_bytes(file_bytes, poppler_path=poppler_path)
            else:
                # Try without explicit path; convert_from_bytes will try to use PATH
                images = convert_from_bytes(file_bytes)
            
            for image in images:
                # Process each page (image) with Tesseract
                raw_text += pytesseract.image_to_string(image, lang='eng', config='--psm 3') + "\n\n"
        
        except Exception as e:
            raise Exception(f"PDF Handling Error: Poppler/PDF2Image failed. Ensure Poppler is installed. Details: {e}")
            
    else:  # Process as a standard image (PNG, JPEG)
        try:
            image = Image.open(BytesIO(file_bytes))
            raw_text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
        
        except Exception as e:
            raise Exception(f"Image Reading Error: Pillow/Tesseract failed. Details: {e}")

    if not raw_text.strip():
        raise Exception("OCR failed to extract any text from the document.")
        
    return raw_text

@app.route('/read-document', methods=['POST'])
def read_document():
    """Handle document upload, OCR, LLM simplification, and translation."""
    if 'document' not in request.files:
        return jsonify({'error': 'No document file uploaded'}), 400

    document_file = request.files['document']
    target_lang = request.form.get('language_code', DEFAULT_LANG)
    use_pdf_processor = request.form.get('use_pdf_processor', 'true').lower() == 'true'  # Default to true for PDFs

    if document_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read file content into memory
    file_bytes = document_file.read()
    
    try:
        # For PDFs, try PDF processor first (if enabled or default)
        if document_file.content_type == 'application/pdf' and use_pdf_processor:
            try:
                # Try to import PDF processor
                try:
                    from pdf_processor import PDFDocumentProcessor
                except ImportError as import_err:
                    error_details = str(import_err)
                    logging.warning(f"PDF processor module not available: {error_details}")
                    # Check if it's a missing dependency
                    if 'fitz' in error_details or 'PyMuPDF' in error_details:
                        raise Exception("PyMuPDF not installed. Install with: pip install PyMuPDF")
                    elif 'transformers' in error_details:
                        raise Exception("Transformers not installed. Install with: pip install transformers torch")
                    else:
                        raise Exception(f"PDF processor import failed: {error_details}")
                
                import tempfile
                import os
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(file_bytes)
                    tmp_path = tmp_file.name
                
                processor = None
                try:
                    # Process PDF with advanced processor
                    logging.info(f"Initializing PDF processor for file: {tmp_path}")
                    try:
                        processor = PDFDocumentProcessor(tmp_path)
                    except Exception as init_err:
                        logging.error(f"Failed to initialize PDF processor: {str(init_err)}")
                        raise Exception(f"PDF processor initialization failed: {str(init_err)}")
                    
                    logging.info("Processing PDF document...")
                    try:
                        results = processor.process_document()
                    except Exception as proc_doc_err:
                        logging.error(f"Error during document processing: {str(proc_doc_err)}")
                        import traceback
                        logging.error(traceback.format_exc())
                        raise Exception(f"Document processing failed: {str(proc_doc_err)}")
                    
                    # Get content overview
                    try:
                        content_overview = processor.get_content_overview()
                        logging.info(f"PDF processed: {results.get('headings', 0)} headings found")
                    except Exception as overview_err:
                        logging.warning(f"Could not get content overview: {str(overview_err)}")
                        content_overview = []
                    
                    # Generate overall summary from first heading if available (optional, don't fail if it errors)
                    overall_summary = ""
                    if content_overview and len(content_overview) > 0:
                        try:
                            first_heading = content_overview[0]['heading']
                            logging.info(f"Summarizing first heading: {first_heading}")
                            overall_summary = processor.summarize_heading(first_heading) or ""
                        except Exception as e:
                            logging.warning(f"Could not summarize first heading: {str(e)}")
                            # Continue without summary
                    
                    # Get raw text from indexed lines before closing
                    raw_text_sample = ""
                    try:
                        if hasattr(processor, 'indexed_lines') and processor.indexed_lines:
                            raw_text_sample = "\n".join([line.get('text', '') for line in processor.indexed_lines[:100] if line.get('text', '').strip()])
                        if not raw_text_sample:
                            raw_text_sample = "Text extraction completed successfully."
                    except Exception as e:
                        logging.warning(f"Could not extract raw text sample: {str(e)}")
                        raw_text_sample = "Text extraction completed successfully."
                    
                    # Prepare response before closing
                    response_data = {
                        "raw_text": raw_text_sample,
                        "english_explanation": overall_summary or "PDF processed successfully. Use content_overview to explore sections.",
                        "vernacular_explanation": overall_summary or "PDF processed successfully.",
                        "content_overview": content_overview,
                        "statistics": {
                            "total_headings": results.get('headings', 0) if results else 0,
                            "total_lines": results.get('indexed_lines', 0) if results else 0
                        },
                        "processed_with": "pdf_processor"
                    }
                    
                    # Clean up processor
                    try:
                        if processor:
                            processor.close()
                    except Exception as close_err:
                        logging.warning(f"Error closing processor: {str(close_err)}")
                    processor = None
                    
                    return jsonify(response_data)
                except Exception as proc_err:
                    logging.error(f"Error in PDF processing: {str(proc_err)}")
                    import traceback
                    logging.error(traceback.format_exc())
                    # Clean up processor if it exists
                    if processor:
                        try:
                            processor.close()
                        except:
                            pass
                    # Don't raise - fall through to OCR fallback
                    logging.info("Falling back to OCR method due to PDF processor error")
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
            except Exception as e:
                error_msg = str(e)
                logging.warning(f"PDF processor failed, falling back to OCR: {error_msg}")
                import traceback
                logging.error(traceback.format_exc())
                # Always fall through to OCR method as fallback
        
        # --- 1. Perform OCR to extract text (using Tesseract) ---
        raw_text = perform_ocr(file_bytes, document_file.content_type) 
        
        if not raw_text.strip():
            return jsonify({'error': 'Could not extract text from the document. Please ensure the image/PDF is clear.'}), 400

        # --- 2. LLM Simplification (Groq) ---
        chatbot = get_chatbot("document_reader")
        if chatbot is None:
            return jsonify({"error": "Chatbot service is unavailable. API keys might be missing."}), 503

        system_prompt_llm = f"""You are an expert educational explainer and tutor, working in 'Explain Like I'm 18 Mode'. Your task is to analyze the following educational document text (which could be a syllabus, course material, textbook content, or academic document).
        
        **Your output MUST be formatted using standard Markdown syntax (e.g., ## for headings, * for lists, ** for bolding) to ensure clarity.**
        
        ## 📚 Summary of Key Educational Concepts
        1. Summarize the **Key Concepts and Topics** covered in the document in a bulleted list, using simple analogies and culturally relevant examples from Indian context.
        
        ## 📖 Learning Objectives and Requirements Explained
        2. Provide a **simple explanation** of the main learning objectives, requirements, and what students need to know or do.
        
        3. Convert all academic jargon into plain-language and relatable examples using Indian cultural context (Indian festivals, traditions, local examples, Indian history, geography, etc.).
        
        4. Explain concepts with step-by-step breakdowns and multiple examples that Indian students can relate to.
        
        5. The final response must be in English for the next translation step.
        
        Document Text:
        ---
        {raw_text[:4000]}
        ---
        """

        chat_completion = chatbot.groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt_llm}],
            model=GROQ_MODEL_CHAT
        )

        english_explanation = chat_completion.choices[0].message.content
        
        if not english_explanation.strip():
            raise Exception("LLM failed to generate an explanation.")

        # --- 3. Translate to Target Vernacular Language (Sarvam AI) ---
        vernacular_explanation = english_explanation
        if target_lang != "en-IN":
            vernacular_explanation = chatbot._translate(
                english_explanation,
                source_lang="en-IN",
                target_lang=target_lang
            )

        # --- 4. Return the result ---
        return jsonify({
            "raw_text": raw_text,
            "english_explanation": english_explanation,
            "vernacular_explanation": vernacular_explanation
        })

    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error in document reader: {error_msg}")
        import traceback
        error_trace = traceback.format_exc()
        logging.error(f"Full traceback:\n{error_trace}")
        
        # Return detailed error for debugging
        return jsonify({
            "error": f"Failed to process document: {error_msg}",
            "error_type": type(e).__name__,
            "hint": "Check Flask server logs for detailed error information. Ensure all dependencies are installed (PyMuPDF, torch, transformers for PDF processor)."
        }), 500

@app.route('/api/summarize-pdf', methods=['POST'])
def summarize_pdf():
    """PDF processing: Extract text, summarize, and generate Q&A from PDF content"""
    logging.info("=== PDF Summarization Request Received ===")
    
    # Initialize response variables
    summary = ""
    qa_pairs = []
    tmp_path = None
    processor = None
    file_bytes = None
    
    try:
        # Validate request
        if 'document' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No document file uploaded',
                'summary': 'Error: No document file uploaded. Please select a PDF file.',
                'questions_and_answers': []
            }), 400

        document_file = request.files['document']
        if document_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No selected file',
                'summary': 'Error: No file selected. Please select a PDF file.',
                'questions_and_answers': []
            }), 400
        
        if document_file.content_type != 'application/pdf':
            return jsonify({
                'success': False,
                'error': 'Only PDF files are supported',
                'summary': 'Error: Only PDF files are supported. Please upload a PDF file.',
                'questions_and_answers': []
            }), 400

        file_bytes = document_file.read()
        logging.info(f"PDF file received: {len(file_bytes)} bytes, filename: {document_file.filename}")
        
        # Import required modules
        from pdf_processor import PDFDocumentProcessor
        import tempfile
        import os
            
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        logging.info(f"Saved PDF to temp file: {tmp_path}")
        
        # Process PDF
        try:
            logging.info("Initializing PDF processor...")
            processor = PDFDocumentProcessor(tmp_path)
            logging.info("Processing document...")
            processor.process_document()
            logging.info("Document processed successfully")
        except Exception as proc_init_err:
            logging.error(f"Error initializing/processing PDF: {proc_init_err}")
            # Try direct extraction as fallback
            try:
                import fitz
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                text_parts = []
                for page_num in range(min(5, len(doc))):
                    page = doc[page_num]
                    text = page.get_text()
                    if text and len(text.strip()) > 20:
                        text_parts.append(text.strip())
                doc.close()
                
                if text_parts:
                    all_text = ' '.join(text_parts)
                    summary = all_text[:1000] + "..." if len(all_text) > 1000 else all_text
                    return jsonify({
                        "success": True,
                        "summary": summary,
                        "questions_and_answers": [],
                        "warning": "Extracted using direct method"
                    })
            except Exception as fallback_err:
                logging.error(f"Fallback extraction also failed: {fallback_err}")
            
            # Clean up and return error
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
            return jsonify({
                "success": False,
                "error": f"Failed to process PDF: {str(proc_init_err)}",
                "summary": f"Error: Could not process PDF. {str(proc_init_err)}",
                "questions_and_answers": []
            }), 500
        
        # Get all text - use multiple methods to ensure we get everything
        all_text = ""
        try:
            all_text = processor.get_all_text()
            logging.info(f"Extracted text via get_all_text: {len(all_text)} characters")
        except Exception as text_err:
            logging.error(f"Error in get_all_text: {text_err}")
            all_text = ""
        
        # Fallback 1: Get from indexed lines (all content lines)
        if not all_text or len(all_text.strip()) < 50:
            logging.warning("Text too short, trying indexed lines extraction...")
            try:
                if hasattr(processor, 'indexed_lines') and processor.indexed_lines:
                    # Get ALL lines, not just content classification
                    all_lines = [line.get('text', '') for line in processor.indexed_lines if line.get('text', '').strip()]
                    all_text = ' '.join(all_lines)
                    logging.info(f"Fallback extraction from indexed_lines: {len(all_text)} characters")
            except Exception as fallback_err:
                logging.error(f"Fallback text extraction failed: {fallback_err}")
        
        # Fallback 2: Get from content mapper using all headings
        if not all_text or len(all_text.strip()) < 50:
            logging.warning("Trying content mapper extraction...")
            try:
                if hasattr(processor, 'content_mapper') and processor.content_mapper:
                    content_parts = []
                    for heading in processor.hierarchy:
                        heading_content = processor.content_mapper.get_content_for_heading(heading)
                        if heading_content and len(heading_content.strip()) > 20:
                            content_parts.append(heading_content)
                    if content_parts:
                        all_text = ' '.join(content_parts)
                        logging.info(f"Content mapper extraction: {len(all_text)} characters")
            except Exception as mapper_err:
                logging.error(f"Content mapper extraction failed: {mapper_err}")
        
        if not all_text or len(all_text.strip()) < 50:
            logging.error("Insufficient text extracted from all methods")
            if processor:
                try:
                    processor.close()
                except:
                    pass
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            return jsonify({
                "success": False,
                "error": "Could not extract sufficient text from PDF",
                "summary": "The PDF might be image-based or corrupted. Please ensure it contains readable text.",
                "questions_and_answers": []
            }), 400
        
        # Use more text for better summaries and Q&A (up to 10000 chars)
        text_for_processing = all_text[:10000] if len(all_text) > 10000 else all_text
        logging.info(f"Text for processing: {len(text_for_processing)} characters (from {len(all_text)} total)")
        
        # Generate summary using the summarizer - MUST ALWAYS RETURN A SUMMARY
        logging.info("Generating summary using AI model...")
        summary = None
        
        # Try AI summarization first
        try:
            if hasattr(processor, 'summarizer') and processor.summarizer:
                try:
                    summary = processor.summarizer.summarize(text_for_processing, max_length=400, min_length=150)
                    if summary and len(summary.strip()) >= 20:
                        logging.info(f"AI Summary generated: {len(summary)} characters")
                except Exception as ai_err:
                    logging.warning(f"AI summarization failed: {ai_err}")
                    summary = None
        except Exception as e:
            logging.warning(f"Summarizer access error: {e}")
            summary = None
        
        # Fallback 1: Extract key sentences (smart extraction)
        if not summary or len(summary.strip()) < 20:
            logging.info("Using sentence extraction fallback...")
            try:
                # Split into sentences
                sentences = [s.strip() for s in text_for_processing.split('.') if len(s.strip()) > 30]
                if sentences:
                    # Take first 5-7 sentences for a good summary
                    summary = '. '.join(sentences[:7]) + '.' if len(sentences) >= 7 else '. '.join(sentences) + '.'
                    logging.info(f"Sentence extraction summary: {len(summary)} characters")
                else:
                    summary = None
            except Exception as fallback_sum_err:
                logging.error(f"Sentence extraction error: {fallback_sum_err}")
                summary = None
        
        # Fallback 2: Use first 500-800 characters
        if not summary or len(summary.strip()) < 20:
            logging.info("Using character extraction fallback...")
            try:
                # Take first meaningful portion
                summary = text_for_processing[:800].strip()
                # Try to end at a sentence boundary
                last_period = summary.rfind('.')
                if last_period > 400:  # If we have a good sentence boundary
                    summary = summary[:last_period + 1]
                else:
                    summary = summary[:500] + "..."
                logging.info(f"Character extraction summary: {len(summary)} characters")
            except Exception as char_err:
                logging.error(f"Character extraction error: {char_err}")
                summary = text_for_processing[:500] + "..."
        
        # Fallback 3: Absolute minimum - just return text
        if not summary or len(summary.strip()) < 10:
            summary = text_for_processing[:500] + "..." if len(text_for_processing) > 500 else text_for_processing
            logging.warning("Using absolute minimum fallback summary")
        
        # Final safety check - MUST have a summary
        summary = str(summary).strip() if summary else text_for_processing[:500] + "..."
        if len(summary) < 20:
            summary = text_for_processing[:500] + "..." if len(text_for_processing) > 500 else "PDF content extracted successfully."
        
        logging.info(f"Final summary length: {len(summary)} characters")
        
        # Generate Q&A from PDF content - STRICT: Only from PDF (optional, don't fail if errors)
        qa_pairs = []
        try:
            chatbot = get_chatbot("pdf_summarizer")
            if chatbot and hasattr(chatbot, 'groq_client') and chatbot.groq_client and text_for_processing:
                logging.info("Generating questions from PDF...")
                
                # Generate questions
                questions_prompt = f"""Based on this PDF content, generate exactly 5 important questions that help understand the key concepts.

PDF Content:
{text_for_processing[:5000]}

Generate exactly 5 questions, one per line, numbered 1-5. Only output the questions, nothing else."""
                
                try:
                    questions_completion = chatbot.groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": questions_prompt}],
                        model=GROQ_MODEL_CHAT
                    )
                    questions_text = questions_completion.choices[0].message.content.strip()
                    logging.info(f"Questions generated: {len(questions_text)} characters")
                except Exception as q_err:
                    logging.error(f"Error generating questions: {q_err}")
                    import traceback
                    logging.error(traceback.format_exc())
                    questions_text = ""
                
                # Parse questions
                questions = []
                if questions_text:
                    for line in questions_text.split('\n'):
                        line = line.strip()
                        if line and len(line) > 5:
                            # Remove numbering
                            if line[0].isdigit():
                                question = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                            elif line.startswith('-') or line.startswith('*'):
                                question = re.sub(r'^[-*]\s*', '', line).strip()
                            else:
                                question = line
                            
                            if question and len(question) > 10:
                                if not question.endswith('?'):
                                    question += '?'
                                questions.append(question)
                
                # Limit to 5 questions
                questions = questions[:5]
                logging.info(f"Parsed {len(questions)} questions")
                
                # Generate answers STRICTLY from PDF content - use more context
                logging.info("Generating answers from PDF content...")
                for idx, question in enumerate(questions, 1):
                    try:
                        logging.info(f"Generating answer {idx}/{len(questions)} for: {question[:50]}...")
                        
                        # Find relevant context for the question (search for keywords in PDF)
                        question_keywords = [w.lower() for w in question.split() if len(w) > 3 and w.lower() not in ['what', 'when', 'where', 'which', 'whose', 'about', 'from', 'this', 'that', 'these', 'those']]
                        
                        # Use full text for better context (up to 8000 chars)
                        context_text = text_for_processing[:8000] if len(text_for_processing) > 8000 else text_for_processing
                        
                        # STRICT prompt - answer ONLY from PDF with emphasis on accuracy
                        answer_prompt = f"""You are a PDF content analyzer. Your task is to answer the question using ONLY the information provided in the PDF content below.

CRITICAL RULES - FOLLOW STRICTLY:
1. Answer ONLY using information from the PDF content provided below
2. Do NOT use any external knowledge, general knowledge, or information not in the PDF
3. If the answer is not in the PDF content, you MUST respond with: "This information is not available in the PDF."
4. Quote directly from the PDF when possible - use exact phrases from the PDF
5. Be precise and accurate - only state facts that appear in the PDF
6. If you're unsure, say "This information is not available in the PDF."
7. Do not make assumptions or inferences beyond what is explicitly stated

PDF Content:
{context_text}

Question: {question}

IMPORTANT: Answer using ONLY the PDF content above. If the answer is not in the PDF, say "This information is not available in the PDF."

Answer:"""
                        
                        answer_completion = chatbot.groq_client.chat.completions.create(
                            messages=[{"role": "user", "content": answer_prompt}],
                            model=GROQ_MODEL_CHAT,
                            temperature=0.1  # Lower temperature for more accurate, factual answers
                        )
                        
                        answer = answer_completion.choices[0].message.content.strip()
                        
                        # Verify answer contains information (not just "not available")
                        if answer and len(answer) > 10:
                            # Check if answer seems to be from PDF (contains some keywords from question or PDF)
                            answer_lower = answer.lower()
                            has_relevant_content = any(
                                keyword in answer_lower or 
                                answer_lower not in ['this information is not available in the pdf.', 'this information is not available in the pdf']
                                for keyword in question_keywords[:3]  # Check first 3 keywords
                            ) if question_keywords else True
                            
                            if has_relevant_content or 'not available' in answer_lower:
                                qa_pairs.append({
                                    "question": question,
                                    "answer": answer
                                })
                                logging.info(f"Answer {idx} generated successfully ({len(answer)} chars)")
                            else:
                                logging.warning(f"Answer {idx} seems irrelevant, skipping")
                    except Exception as e:
                        logging.warning(f"Error generating answer for '{question}': {e}")
                        import traceback
                        logging.warning(traceback.format_exc())
                        continue
                
                logging.info(f"Generated {len(qa_pairs)} Q&A pairs")
            else:
                logging.warning("Chatbot not available for Q&A generation")
        except Exception as qa_err:
            logging.warning(f"Q&A generation failed: {qa_err}")
            import traceback
            logging.warning(traceback.format_exc())
            # Continue without Q&A - summary is more important
        
        # Clean up processor
        if processor:
            try:
                processor.close()
            except:
                pass
        
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        # ALWAYS return summary - even if Q&A failed
        if not summary or len(summary.strip()) < 10:
            summary = "PDF processed successfully. Content extracted from the document."
        
        response = {
            "success": True,
            "summary": str(summary),
            "questions_and_answers": qa_pairs if qa_pairs else []
        }
        logging.info(f"=== SUCCESS: Returning response with summary ({len(summary)} chars) and {len(qa_pairs)} Q&A pairs ===")
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"PDF processing error: {str(e)}")
        import traceback
        error_trace = traceback.format_exc()
        logging.error(f"Full traceback:\n{error_trace}")
        
        # Last resort: Try to extract text directly from PDF bytes
        fallback_summary = None
        try:
            import fitz  # PyMuPDF
            if file_bytes:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                text_parts = []
                for page_num in range(min(5, len(doc))):
                    page = doc[page_num]
                    text = page.get_text()
                    if text and len(text.strip()) > 20:
                        text_parts.append(text.strip())
                doc.close()
                
                if text_parts:
                    combined_text = ' '.join(text_parts)
                    sentences = [s.strip() for s in combined_text.split('.') if len(s.strip()) > 30]
                    if sentences:
                        fallback_summary = '. '.join(sentences[:5]) + '.'
                    else:
                        fallback_summary = combined_text[:500] + "..."
                    logging.info(f"Extracted fallback summary via PyMuPDF: {len(fallback_summary)} chars")
        except Exception as pdf_err:
            logging.warning(f"PyMuPDF fallback also failed: {pdf_err}")
        
        # Clean up
        if processor:
            try:
                processor.close()
            except:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        # ALWAYS return something
        if fallback_summary:
            return jsonify({
                "success": True,
                "summary": fallback_summary,
                "questions_and_answers": [],
                "warning": "Extracted using fallback method"
            })
        
        return jsonify({
            "success": False,
            "error": f"Failed to process PDF: {str(e)}",
            "summary": f"Error: Could not process PDF. {str(e)}. Please ensure the PDF contains readable text and try again.",
            "questions_and_answers": []
        }), 500

@app.route('/api/summarize-section', methods=['POST'])
def summarize_section():
    """Summarize a specific section (heading or subheading) from a processed PDF"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    pdf_path = data.get('pdf_path')
    section_text = data.get('section_text')
    section_type = data.get('section_type', 'subheading')  # 'heading' or 'subheading'
    
    if not pdf_path or not section_text:
        return jsonify({'error': 'pdf_path and section_text are required'}), 400
    
    try:
        from pdf_processor import PDFDocumentProcessor
        import os
        
        if not os.path.exists(pdf_path):
            return jsonify({'error': 'PDF file not found'}), 404
        
        processor = PDFDocumentProcessor(pdf_path)
        processor.process_document()
        
        # Summarize based on type
        if section_type == 'heading':
            summary = processor.summarize_heading(section_text)
        else:
            summary = processor.summarize_subheading(section_text)
        
        processor.close()
        
        if summary:
            return jsonify({
                "success": True,
                "section_text": section_text,
                "section_type": section_type,
                "summary": summary
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Section '{section_text}' not found"
            }), 404

    except Exception as e:
        logging.error(f"Error summarizing section: {str(e)}")
        return jsonify({"error": f"Failed to summarize section: {str(e)}"}), 500

@app.route('/api/pdf-qa', methods=['POST'])
def pdf_qa():
    """Answer questions about a PDF using exact content from the document"""
    if 'document' not in request.files:
        return jsonify({'error': 'No document file uploaded'}), 400
    
    question = request.form.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    document_file = request.files['document']
    if document_file.filename == '' or document_file.content_type != 'application/pdf':
        return jsonify({'error': 'Invalid PDF file'}), 400
    
    file_bytes = document_file.read()
    
    try:
        from pdf_processor import PDFDocumentProcessor
        import tempfile
        import os
        import re
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        
        processor = None
        try:
            # Process PDF
            processor = PDFDocumentProcessor(tmp_path)
            processor.process_document()
            
            # Search through indexed lines for relevant content
            question_lower = question.lower()
            question_words = set(re.findall(r'\b\w+\b', question_lower))
            
            # Find relevant lines by keyword matching
            relevant_lines = []
            for line in processor.indexed_lines:
                line_text = line.get('text', '').lower()
                line_words = set(re.findall(r'\b\w+\b', line_text))
                
                # Calculate relevance score (word overlap)
                overlap = len(question_words.intersection(line_words))
                if overlap > 0:
                    relevant_lines.append({
                        'text': line.get('text', ''),
                        'page': line.get('page', 0),
                        'line_number': line.get('line_number', 0),
                        'relevance': overlap
                    })
            
            # Sort by relevance and get top matches
            relevant_lines.sort(key=lambda x: x['relevance'], reverse=True)
            top_lines = relevant_lines[:20]  # Get top 20 most relevant lines
            
            # Combine relevant content
            context_text = "\n".join([
                f"[Page {line['page']}, Line {line['line_number']}]: {line['text']}"
                for line in top_lines
            ])
            
            if not context_text.strip():
                # Fallback: use all content lines
                context_text = "\n".join([
                    f"[Page {line.get('page', 0)}, Line {line.get('line_number', 0)}]: {line.get('text', '')}"
                    for line in processor.indexed_lines[:100]
                ])
            
            # Use chatbot to answer based on the extracted context
            chatbot = get_chatbot("pdf_qa")
            if chatbot is None:
                return jsonify({"error": "Chatbot service is unavailable. API keys might be missing."}), 503
            
            # Create prompt that emphasizes using only the provided context
            system_prompt = f"""You are a helpful assistant that answers questions based EXACTLY on the provided PDF content. 
            
IMPORTANT RULES:
1. Answer ONLY using information from the provided PDF content below
2. If the answer is not in the provided content, say "I cannot find this information in the PDF"
3. Quote specific page and line numbers when referencing content
4. Be precise and cite your sources from the PDF

PDF Content:
{context_text[:4000]}

Question: {question}

Answer based ONLY on the PDF content above:"""
            
            chat_completion = chatbot.groq_client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}],
                model=GROQ_MODEL_CHAT
            )
            
            answer = chat_completion.choices[0].message.content
            
            # Include source references
            source_references = []
            if top_lines:
                pages_mentioned = sorted(set(line['page'] for line in top_lines[:5]))
                source_references = [f"Page {p}" for p in pages_mentioned]
            
            processor.close()
            processor = None
            
            return jsonify({
                "success": True,
                "question": question,
                "answer": answer,
                "sources": source_references,
                "relevant_lines_found": len(relevant_lines)
            })
            
        except Exception as proc_err:
            logging.error(f"Error in PDF Q&A: {str(proc_err)}")
            import traceback
            logging.error(traceback.format_exc())
            if processor:
                try:
                    processor.close()
                except:
                    pass
            raise proc_err
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    except Exception as e:
        logging.error(f"Error in PDF Q&A: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to answer question: {str(e)}"}), 500
if __name__ == '__main__':
    logging.info("Starting SARVASVA AI Service...")
    AI_PORT = int(os.getenv('AI_PORT', 5001))
    app.run(host='127.0.0.1', port=AI_PORT, debug=True)