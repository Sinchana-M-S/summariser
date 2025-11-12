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
    const response = await axios.get(`${AI_SERVICE_URL}/`);
    res.json({ status: "connected", aiService: "online" });
  } catch (error) {
    res.status(503).json({ status: "disconnected", aiService: "offline", error: error.message });
  }
});

// ✅ Chat endpoint - proxy to Flask /chat
router.post("/chat", async (req, res) => {
  try {
    const response = await axios.post(`${AI_SERVICE_URL}/chat`, req.body, {
      headers: { "Content-Type": "application/json" },
    });
    res.json(response.data);
  } catch (error) {
    console.error("AI Chat Error:", error.response?.data || error.message);
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "AI service unavailable",
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
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Summarization failed",
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

    const response = await axios.post(`${AI_SERVICE_URL}/read-document`, formData, {
      headers: {
        ...formData.getHeaders(),
      },
    });

    res.json(response.data);
  } catch (error) {
    console.error("Document Read Error:", error.response?.data || error.message);
    res.status(error.response?.status || 500).json({
      error: error.response?.data?.error || "Document processing failed",
    });
  }
});

module.exports = router;

