const express = require("express");
const axios = require("axios");
const multer = require("multer");
const router = express.Router();

// ✅ AI Service URL (Flask backend) - loaded from environment
// This should match the AI_SERVICE_URL in server/.env
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://127.0.0.1:5001";

// ✅ Configure multer for file uploads
const upload = multer({ storage: multer.memoryStorage() });

// ✅ Health check for AI service
router.get("/health", async (req, res) => {
  try {
    const response = await axios.get(`${AI_SERVICE_URL}/`, { timeout: 5000 });
    res.json({ status: "connected", aiService: "online", aiServiceUrl: AI_SERVICE_URL });
  } catch (error) {
    console.error(`AI Service Health Check Failed: ${AI_SERVICE_URL} - ${error.message}`);
    res.status(503).json({ 
      status: "disconnected", 
      aiService: "offline", 
      aiServiceUrl: AI_SERVICE_URL,
      error: error.message,
      hint: "Make sure Flask AI service is running on port 5001"
    });
  }
});

// ✅ Chat endpoint - proxy to Flask /chat
router.post("/chat", async (req, res) => {
  try {
    console.log(`Proxying chat request to ${AI_SERVICE_URL}/chat`);
    const response = await axios.post(`${AI_SERVICE_URL}/chat`, req.body, {
      headers: { "Content-Type": "application/json" },
      timeout: 60000, // 60 second timeout for chat
    });
    res.json(response.data);
  } catch (error) {
    console.error("AI Chat Error:", error.response?.data || error.message);
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: "AI service is not running. Please start the Flask AI service on port 5001.",
        hint: "Run: cd 'ai models' && python main.py"
      });
    }
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "AI service unavailable",
      details: error.message
    });
  }
});

// ✅ Speech to Text - proxy to Flask /speech-to-text
router.post("/speech-to-text", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No audio file uploaded" });
    }

    const FormData = require("form-data");
    const formData = new FormData();
    formData.append("audio", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });
    formData.append("language_code", req.body.language_code || "en-IN");

    const response = await axios.post(`${AI_SERVICE_URL}/speech-to-text`, formData, {
      headers: {
        ...formData.getHeaders(),
      },
    });

    res.json(response.data);
  } catch (error) {
    console.error("STT Error:", error.response?.data || error.message);
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Speech-to-text service unavailable",
    });
  }
});

// ✅ Text to Speech - proxy to Flask /text-to-speech
router.post("/text-to-speech", async (req, res) => {
  try {
    const response = await axios.post(
      `${AI_SERVICE_URL}/text-to-speech`,
      req.body,
      {
        headers: { "Content-Type": "application/json" },
        responseType: "arraybuffer",
      }
    );

    res.setHeader("Content-Type", "audio/mpeg");
    res.send(Buffer.from(response.data));
  } catch (error) {
    console.error("TTS Error:", error.response?.data || error.message);
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Text-to-speech service unavailable",
    });
  }
});

// ✅ Set Language - proxy to Flask /set-language
router.post("/set-language", async (req, res) => {
  try {
    const response = await axios.post(`${AI_SERVICE_URL}/set-language`, req.body, {
      headers: { "Content-Type": "application/json" },
    });
    res.json(response.data);
  } catch (error) {
    console.error("Set Language Error:", error.response?.data || error.message);
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Language setting failed",
    });
  }
});

// ✅ Translate - proxy to Flask /translate
router.post("/translate", async (req, res) => {
  try {
    const response = await axios.post(`${AI_SERVICE_URL}/translate`, req.body, {
      headers: { "Content-Type": "application/json" },
    });
    res.json(response.data);
  } catch (error) {
    console.error("Translation Error:", error.response?.data || error.message);
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Translation failed",
    });
  }
});

// ✅ Generate Quiz - proxy to Flask /api/generate-quiz
router.post("/generate-quiz", async (req, res) => {
  try {
    const response = await axios.post(`${AI_SERVICE_URL}/api/generate-quiz`, req.body, {
      headers: { "Content-Type": "application/json" },
    });
    res.json(response.data);
  } catch (error) {
    console.error("Quiz Generation Error:", error.response?.data || error.message);
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Quiz generation failed",
    });
  }
});

// ✅ Summarize - proxy to Flask /api/summarize
router.post("/summarize", async (req, res) => {
  try {
    const response = await axios.post(`${AI_SERVICE_URL}/api/summarize`, req.body, {
      headers: { "Content-Type": "application/json" },
    });
    res.json(response.data);
  } catch (error) {
    console.error("Summarization Error:", error.response?.data || error.message);
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: "AI service is not running. Please start the Flask AI service on port 5001.",
        hint: "Run: cd 'ai models' && python main.py"
      });
    }
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Summarization failed",
      details: error.message
    });
  }
});

// ✅ Summarize PDF - Advanced PDF processing with ML/DL (like sum folder)
router.post("/summarize-pdf", upload.single("document"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No PDF file uploaded" });
    }

    const FormData = require("form-data");
    const formData = new FormData();
    formData.append("document", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    const response = await axios.post(`${AI_SERVICE_URL}/api/summarize-pdf`, formData, {
      headers: {
        ...formData.getHeaders(),
      },
      timeout: 300000, // 5 minutes timeout for PDF processing
    });

    res.json(response.data);
  } catch (error) {
    console.error("PDF Summarization Error:", error.response?.data || error.message);
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: "AI service is not running. Please start the Flask AI service on port 5001.",
        hint: "Run: cd 'ai models' && python main.py"
      });
    }
    if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
      return res.status(504).json({
        error: "PDF processing timed out. The PDF might be too large or complex.",
        hint: "Try with a smaller PDF or check Flask server logs"
      });
    }
    res.status(error.response?.status || 500).json({
      success: false,
      error: error.response?.data?.error || error.response?.data?.message || "PDF processing failed",
      details: error.response?.data?.details || error.message,
      error_type: error.response?.data?.error_type
    });
  }
});

// ✅ Summarize Section - Summarize a specific heading/subheading
router.post("/summarize-section", async (req, res) => {
  try {
    const response = await axios.post(`${AI_SERVICE_URL}/api/summarize-section`, req.body, {
      headers: { "Content-Type": "application/json" },
    });
    res.json(response.data);
  } catch (error) {
    console.error("Section Summarization Error:", error.response?.data || error.message);
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: "AI service is not running. Please start the Flask AI service on port 5001.",
        hint: "Run: cd 'ai models' && python main.py"
      });
    }
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Section summarization failed",
      details: error.message
    });
  }
});

// ✅ Explain Timestamp - proxy to Flask /api/explain-timestamp
router.post("/explain-timestamp", async (req, res) => {
  try {
    const response = await axios.post(
      `${AI_SERVICE_URL}/api/explain-timestamp`,
      req.body,
      {
        headers: { "Content-Type": "application/json" },
      }
    );
    res.json(response.data);
  } catch (error) {
    console.error("Timestamp Explanation Error:", error.response?.data || error.message);
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Explanation failed",
    });
  }
});

// ✅ Upload Course - proxy to Flask /api/upload-course
router.post("/upload-course", async (req, res) => {
  try {
    const response = await axios.post(`${AI_SERVICE_URL}/api/upload-course`, req.body, {
      headers: { "Content-Type": "application/json" },
    });
    res.json(response.data);
  } catch (error) {
    console.error("Course Upload Error:", error.response?.data || error.message);
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Course upload failed",
    });
  }
});

// ✅ Peer Chat - proxy to Flask /api/peer-chat
router.post("/peer-chat", async (req, res) => {
  try {
    const response = await axios.post(`${AI_SERVICE_URL}/api/peer-chat`, req.body, {
      headers: { "Content-Type": "application/json" },
    });
    res.json(response.data);
  } catch (error) {
    console.error("Peer Chat Error:", error.response?.data || error.message);
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Peer chat failed",
    });
  }
});

// ✅ PDF Q&A - Answer questions about PDF using exact content
router.post("/pdf-qa", upload.single("document"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No PDF file uploaded" });
    }

    const question = req.body.question;
    if (!question || !question.trim()) {
      return res.status(400).json({ error: "No question provided" });
    }

    const FormData = require("form-data");
    const formData = new FormData();
    formData.append("document", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });
    formData.append("question", question);

    const response = await axios.post(`${AI_SERVICE_URL}/api/pdf-qa`, formData, {
      headers: {
        ...formData.getHeaders(),
      },
      timeout: 300000, // 5 minutes timeout
    });

    res.json(response.data);
  } catch (error) {
    console.error("PDF Q&A Error:", error.response?.data || error.message);
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: "AI service is not running. Please start the Flask AI service on port 5001.",
        hint: "Run: cd 'ai models' && python main.py"
      });
    }
    if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
      return res.status(504).json({
        error: "Question answering timed out. The PDF might be too large.",
        hint: "Try with a smaller PDF or check Flask server logs"
      });
    }
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Failed to answer question",
      details: error.response?.data?.details || error.message
    });
  }
});

// ✅ Read Document - proxy to Flask /read-document
router.post("/read-document", upload.single("document"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No document file uploaded" });
    }

    const FormData = require("form-data");
    const formData = new FormData();
    formData.append("document", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });
    
    // Add language code if provided
    if (req.body.language_code) {
      formData.append("language_code", req.body.language_code);
    }

    const response = await axios.post(`${AI_SERVICE_URL}/read-document`, formData, {
      headers: {
        ...formData.getHeaders(),
      },
      timeout: 300000, // 5 minutes timeout for document processing (PDF processing can be slow)
    });

    res.json(response.data);
  } catch (error) {
    console.error("Document Read Error:", error.response?.data || error.message);
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: "AI service is not running. Please start the Flask AI service on port 5001.",
        hint: "Run: cd 'ai models' && python main.py"
      });
    }
    if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
      return res.status(504).json({
        error: "Document processing timed out. The PDF might be too large or complex.",
        hint: "Try with a smaller PDF or check Flask server logs"
      });
    }
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || error.response?.data?.message || "Document processing failed",
      details: error.response?.data?.hint || error.message,
      error_type: error.response?.data?.error_type
    });
  }
});

module.exports = router;

