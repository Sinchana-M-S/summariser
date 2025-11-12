const express = require("express");
const cors = require("cors");
const path = require("path");
require("dotenv").config();

// ✅ Load environment variables
const PORT = process.env.PORT || 5000;
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://127.0.0.1:5001";
const CORS_ORIGINS = process.env.CORS_ORIGINS 
  ? process.env.CORS_ORIGINS.split(",").map(origin => origin.trim())
  : undefined;

// ✅ Routes
const authRoutes = require("./routes/auth.js");
const coursesRoutes = require("./routes/courses.js");
const creditsRoutes = require("./routes/credits.js");
const chatRoutes = require("./routes/chat.js");
const documentsRoutes = require("./routes/documents.js");
const assessmentsRoutes = require("./routes/assessments.js");
const liveClassesRoutes = require("./routes/liveClasses.js");
const usersRoutes = require("./routes/users.js");
const aiRoutes = require("./routes/ai.js"); // ✅ AI proxy routes

const app = express();

// ✅ Middlewares
app.use(cors({
  origin: CORS_ORIGINS || true, // Allow all origins if not specified
  credentials: true
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ✅ Register routes
app.use("/api/auth", authRoutes);
app.use("/api/courses", coursesRoutes);
app.use("/api/credits", creditsRoutes);
app.use("/api/chat", chatRoutes);
app.use("/api/documents", documentsRoutes);
app.use("/api/assessments", assessmentsRoutes);
app.use("/api/live-classes", liveClassesRoutes);
app.use("/api/users", usersRoutes);
app.use("/api/ai", aiRoutes); // ✅ AI routes (proxies to Flask)

// ✅ Test route
app.get("/", (req, res) => {
  res.send("✅ Sarvasva Backend Running (CommonJS)");
});

// ✅ Start server
app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
  console.log(`✅ AI Service URL: ${AI_SERVICE_URL}`);
  if (CORS_ORIGINS) {
    console.log(`✅ CORS Origins: ${CORS_ORIGINS.join(", ")}`);
  }
});
