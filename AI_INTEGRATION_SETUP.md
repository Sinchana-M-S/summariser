# AI Models Integration Setup Guide

This guide explains how to connect the Flask AI service to your Node.js backend and React frontend.

## Architecture Overview

```
Frontend (React) → Node.js Server (Port 5000) → Flask AI Service (Port 5001)
```

The Node.js server acts as a proxy, forwarding AI requests to the Flask service.

## Setup Instructions

### 1. Install Dependencies

#### Node.js Server
```bash
cd server
npm install
```

This will install:
- `axios` - For making HTTP requests to Flask
- `form-data` - For handling multipart/form-data file uploads
- Other existing dependencies

#### Flask AI Service
```bash
cd "ai models"
pip install -r requirements.txt
```

### 2. Configure Environment Variables

#### Node.js Server (`server/.env`)
Create a `.env` file in the `server` folder:
```env
PORT=5000
AI_SERVICE_URL=http://127.0.0.1:5001
```

#### Flask AI Service (`ai models/.env`)
Create a `.env` file in the `ai models` folder:
```env
AI_PORT=5001
SARVAM_API_KEY=your_sarvam_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

**Important:** Replace the API keys with your actual keys from:
- [Sarvam AI](https://sarvam.ai) - For speech-to-text and text-to-speech
- [Groq](https://groq.com) - For LLM (Large Language Model)

### 3. Start the Services

#### Terminal 1: Start Flask AI Service
```bash
cd "ai models"
python main.py
```

You should see:
```
Starting SARVASVA AI Service...
 * Running on http://127.0.0.1:5001
```

#### Terminal 2: Start Node.js Server
```bash
cd server
npm start
```

You should see:
```
✅ Server running on port 5000
```

#### Terminal 3: Start React Frontend
```bash
cd client
npm run dev
```

### 4. Verify Integration

1. **Check AI Service Health:**
   ```bash
   curl http://localhost:5000/api/ai/health
   ```
   Should return: `{"status":"connected","aiService":"online"}`

2. **Test Chat Endpoint:**
   ```bash
   curl -X POST http://localhost:5000/api/ai/chat \
     -H "Content-Type: application/json" \
     -d '{"message":"Hello","session_id":"test123"}'
   ```

3. **Test in Frontend:**
   - Navigate to `/ai` page
   - Type a message and send
   - Should receive AI response

## Available AI Endpoints

All endpoints are prefixed with `/api/ai`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ai/health` | GET | Check if AI service is running |
| `/api/ai/chat` | POST | Chat with AI (text input) |
| `/api/ai/speech-to-text` | POST | Convert audio to text |
| `/api/ai/text-to-speech` | POST | Convert text to audio |
| `/api/ai/set-language` | POST | Set language for AI responses |
| `/api/ai/generate-quiz` | POST | Generate quiz from topic |
| `/api/ai/summarize` | POST | Summarize text/course/video |
| `/api/ai/explain-timestamp` | POST | Explain video at timestamp |
| `/api/ai/upload-course` | POST | Upload course for processing |
| `/api/ai/peer-chat` | POST | Peer-to-peer chat (placeholder) |
| `/api/ai/read-document` | POST | Process document (OCR) |

## Frontend Integration

### AI Chat Page (`/ai`)
- Uses `/api/ai/chat` for text messages
- Uses `/api/ai/speech-to-text` for voice input

### Student Summarizer (`/student/summarizer`)
- Uses `/api/ai/summarize` for generating summaries

### Course Player
- Can use `/api/ai/explain-timestamp` for video explanations
- Can use `/api/ai/summarize` for video summaries

## Troubleshooting

### Issue: "AI Server Error" in frontend
**Solution:**
1. Check if Flask service is running on port 5001
2. Verify API keys in `ai models/.env`
3. Check Node.js server logs for errors

### Issue: "AI service unavailable" 
**Solution:**
1. Ensure Flask service is started before Node.js server
2. Check `AI_SERVICE_URL` in `server/.env` matches Flask port
3. Test Flask directly: `curl http://127.0.0.1:5001/`

### Issue: Port conflicts
**Solution:**
- Node.js server uses port 5000 (change in `server/.env`)
- Flask uses port 5001 (change in `ai models/.env`)
- Update `AI_SERVICE_URL` if you change Flask port

### Issue: File uploads not working
**Solution:**
- Ensure `form-data` package is installed in Node.js server
- Check multer configuration in `server/routes/ai.js`

## Development Notes

- The Node.js server proxies all requests to Flask, so you only need to configure the frontend to use `http://localhost:5000/api/ai/*`
- Flask service must be running for AI features to work
- API keys are required for production use
- For development/testing, you may need to mock some endpoints if API keys are not available

## Next Steps

1. Add authentication middleware to AI routes if needed
2. Implement rate limiting for AI endpoints
3. Add caching for frequently requested AI responses
4. Set up error monitoring and logging
5. Configure production environment variables

