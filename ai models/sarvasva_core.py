from groq import Groq 
import requests
import logging
import base64
import os
import re
from io import BytesIO
import json
from typing import Dict, Any, List
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
GROQ_MODEL = "llama-3.1-8b-instant"
TTS_API_URL = "https://api.sarvam.ai/text-to-speech"
TRANSLATE_API_URL = "https://api.sarvam.ai/translate"

TTS_CONFIGS = {
    'en-IN': {"model": "bulbul:v2", "speaker": "anushka", "chunk_size": 500},
    'hi-IN': {"model": "bulbul:v2", "speaker": "abhilash", "chunk_size": 300},
    'ta-IN': {"model": "bulbul:v2", "speaker": "vidya", "chunk_size": 300},
    'bn-IN': {"model": "bulbul:v2", "speaker": "ishita", "chunk_size": 300},
    'gu-IN': {"model": "bulbul:v2", "speaker": "kiran", "chunk_size": 300},
    'kn-IN': {"model": "bulbul:v2", "speaker": "kavya", "chunk_size": 300},
    'ml-IN': {"model": "bulbul:v2", "speaker": "arya", "chunk_size": 300},
    'mr-IN': {"model": "bulbul:v2", "speaker": "sakshi", "chunk_size": 300},
    'od-IN': {"model": "bulbul:v2", "speaker": "diya", "chunk_size": 300},
    'pa-IN': {"model": "bulbul:v2", "speaker": "ranjit", "chunk_size": 300},
    'te-IN': {"model": "bulbul:v2", "speaker": "teja", "chunk_size": 300},
    
}

class SarvasvaChatbot:
    def __init__(self, groq_api_key: str, sarvam_api_key: str, language_code: str = "en-IN"):
        if not groq_api_key or not sarvam_api_key:
            raise ValueError("API keys must be provided to SarvasvaChatbot.")
        self.groq_client = Groq(api_key=groq_api_key)
        self.sarvam_key = sarvam_api_key
        self.model_name = GROQ_MODEL
        self.language_code = language_code
        self.history: List[Dict[str, str]] = []
        self.system_prompt = self._get_system_prompt()
        self.reset_session()
    def _get_system_prompt(self):
        return ("You are a friendly, expert, and empathetic educational assistant named SARVASVA. "
                "Your goal is to answer the user's questions clearly, concisely, and accurately "
                "in the context of skill development, science, and general education. "
                "Your responses MUST be in English for internal processing, but avoid mentioning that you are translating.")
    def set_language(self, language_code: str):
        """Changes the current language code for I/O."""
        self.language_code = language_code
    def reset_session(self):
        """Clears the history to start a new conversation (persists the system prompt)."""
        self.history = [{"role": "system", "content": self.system_prompt}]
    def _chunk_text_by_sentence(self, input_text: str, max_length: int) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', input_text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk: chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk: chunks.append(current_chunk.strip())
        return chunks
    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang == target_lang: return text
        chunks = self._chunk_text_by_sentence(text, max_length=950)
        translated_chunks = []
        headers = {"Content-Type": "application/json", "api-subscription-key": self.sarvam_key}
        for chunk in chunks:
            payload = {
                "input": chunk,
                "source_language_code": source_lang,
                "target_language_code": target_lang,
                "mode": "formal",
                "model": "mayura:v1",
            }
            try:
                response = requests.post(TRANSLATE_API_URL, json=payload, headers=headers, timeout=15)
                response.raise_for_status()
                data = response.json()
                translated_chunks.append(data.get("translated_text", chunk))
            except Exception as e:
                logging.error(f"Translation failed for chunk: {e}")
                translated_chunks.append(chunk)
        return " ".join(translated_chunks)
    def _tts(self, text: str, target_lang: str) -> str | None:
        config = TTS_CONFIGS.get(target_lang, TTS_CONFIGS['en-IN'])
        model = config["model"]
        speaker = config["speaker"]
        chunk_size = config["chunk_size"]
        audio_data_combined = BytesIO()
        text_chunks = self._chunk_text_by_sentence(text, max_length=chunk_size)
        headers = {"api-subscription-key": self.sarvam_key, "Content-Type": "application/json"}
        for chunk in text_chunks:
            if not chunk.strip(): continue
            request_body = {
                "inputs": [chunk],
                "target_language_code": target_lang,
                "speaker": speaker,
                "model": model
            }
            try:
                response = requests.post(TTS_API_URL, headers=headers, json=request_body, timeout=15)
                response.raise_for_status()
                result = response.json()
                if "audios" in result and result["audios"]:
                    audio_data_combined.write(base64.b64decode(result["audios"][0]))
            except Exception as e:
                logging.error(f"TTS chunk failure: {e}")
                continue
        if audio_data_combined.getbuffer().nbytes == 0:
            return None
        return base64.b64encode(audio_data_combined.getvalue()).decode('utf-8')
    def explain_timestamp_doubt(self, course_topic: str, video_transcript: str, timestamp: str, user_doubt: str, target_lang: str) -> Dict[str, Any]:
        """
        Instructs the LLM to act as an expert and explain content around a specific timestamp (Feature 2).
        """
        explanation_prompt = f"""
        You are an expert tutor for a course titled '{course_topic}'. 
        A student has paused the video at {timestamp} and has asked a question based on the following transcript segment.
        Analyze the transcript and provide a clear, concise, and helpful explanation that directly addresses the student's doubt. Use simple language and maintain an empathetic, teaching tone.
        Student Doubt: "{user_doubt}"
        Transcript Segment (Context around {timestamp}):
        ---
        {video_transcript}
        ---
        Provide your full explanation in English.
        """
        try:
            completion = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt().replace("educational assistant", "expert course tutor")},
                    {"role": "user", "content": explanation_prompt}
                ]
            )
            english_explanation = completion.choices[0].message.content
            vernacular_explanation = self._translate(
                english_explanation, 
                source_lang="en-IN", 
                target_lang=target_lang
            )
            base64_audio = self._tts(vernacular_explanation, target_lang)
            return {
                "text": vernacular_explanation,
                "audio_base64": base64_audio,
                "language_code": target_lang
            }
        except Exception as e:
            logging.error(f"Timestamp Explanation failed: {e}")
            return {
                "text": f"Error: Could not provide explanation. Details: {str(e)}",
                "audio_base64": None,
                "language_code": target_lang
            }
    def generate_quiz(self, topic: str) -> str:
        """Instructs the LLM to generate a structured JSON quiz (Feature 6)."""
        quiz_prompt = f"""
        Generate a concise, educational quiz based on the following topic: "{topic}".
        The output MUST strictly be a JSON array of 5 questions. Do not include any introductory text, apologies, or markdown formatting outside of the JSON block.
        """
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": quiz_prompt}],
                response_model="json" 
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Quiz generation failed: {e}")
            return json.dumps([{"error": "Failed to generate quiz: LLM or Schema error."}])
    def summarize_resource(self, resource_text: str, target_lang: str) -> str:
        """Instructs the LLM to summarize a resource and translates the output (Feature 4)."""
        summary_prompt = f"""
        You are an educational summarization expert. Condense the following course material 
        into 5 concise bullet points, focusing only on the core concepts a student needs to know. 
        Use a professional, clear tone.
        Resource:
        ---
        {resource_text[:6000]} 
        ---
        """
        try:
            completion = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": summary_prompt}]
            )
            english_summary = completion.choices[0].message.content
            final_summary = self._translate(
                english_summary, source_lang="en-IN", target_lang=target_lang
            )
            return final_summary
        except Exception as e:
            logging.error(f"Summarization failed: {e}")
            return f"Error: Failed to generate summary. Details: {e}"
    def generate_response(self, user_message_vernacular: str, is_voice_chat: bool = False) -> Dict[str, Any]:
        """
        Processes user input, generates an LLM response with history (memory), 
        and prepares the output (text and optional audio).
        """
        current_lang = self.language_code
        english_input = self._translate(
            user_message_vernacular, 
            source_lang=current_lang, 
            target_lang="en-IN"
        )
        self.history.append({"role": "user", "content": english_input})
        try:
            completion = self.groq_client.chat.completions.create(
                model=self.model_name, 
                messages=self.history 
            )
            english_output = completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq LLM generation failed: {e}")
            english_output = "I am sorry, I ran into a system error. Please check the server connection."
        self.history.append({"role": "assistant", "content": english_output})
        vernacular_output_text = self._translate(
            english_output, 
            source_lang="en-IN", 
            target_lang=current_lang
        )
        base64_audio = None
        if is_voice_chat:
            base64_audio = self._tts(vernacular_output_text, current_lang)
        return {
            "text": vernacular_output_text,
            "audio_base64": base64_audio,
            "is_voice_mode": is_voice_chat,
            "history_length": len(self.history)
        }