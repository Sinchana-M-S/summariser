import axios from "axios";

// Use Node.js backend API URL (which proxies to Flask AI service)
const AI_API_BASE = import.meta.env.VITE_API_URL || "http://localhost:5000";

const AI = axios.create({
  baseURL: `${AI_API_BASE}/api/ai`,
});

export default AI;
