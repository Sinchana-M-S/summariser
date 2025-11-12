import { useState, useEffect, useRef } from "react";
import { useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { Send, Upload, User } from "lucide-react";
import { io } from "socket.io-client";
import api from "../../lib/api";
import { Card, CardContent } from "../../components/ui/Card";
import Button from "../../components/ui/Button";
import toast from "react-hot-toast";

export default function StudentChat() {
  const { courseId } = useParams();

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [socket, setSocket] = useState(null);

  const messagesEndRef = useRef(null);

  // ✅ Guest identity for chat
  const guestId =
    localStorage.getItem("guestId") ||
    `g-${Math.random().toString(36).substring(2, 10)}`;
  const guestName =
    localStorage.getItem("guestName") ||
    `Guest-${Math.floor(Math.random() * 9000 + 1000)}`;

  useEffect(() => {
    localStorage.setItem("guestId", guestId);
    localStorage.setItem("guestName", guestName);
  }, []);

  // ✅ Fetch previous messages + connect socket
  useEffect(() => {
    fetchMessages();
    connectSocket();

    return () => {
      if (socket) socket.disconnect();
    };
  }, [courseId]);

  // ✅ Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ✅ Load chat history from JSON backend
  const fetchMessages = async () => {
    try {
      const { data } = await api.get(
        `/chat/course/${courseId || "default"}`,
        {
          headers: {
            "x-guest-id": guestId,
            "x-guest-name": guestName,
          },
        }
      );

      setMessages(data.messages || []);
    } catch (error) {
      console.error("Failed to load messages:", error);
    }
  };

  // ✅ Connect to socket.io
  const connectSocket = () => {
    const SOCKET_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
    const newSocket = io(SOCKET_URL, {
      transports: ["websocket"],
    });

    newSocket.on("connect", () => {
      newSocket.emit("join-course", courseId || "default");
    });

    newSocket.on("new-message", (msg) => {
      setMessages((prev) => [...prev, msg]);
    });

    setSocket(newSocket);
  };

  // ✅ Send message (guest mode)
  const handleSend = async () => {
    if (!input.trim()) return;

    try {
      const { data } = await api.post(
        `/chat/course/${courseId || "default"}/message`,
        { message: input },
        {
          headers: {
            "x-guest-id": guestId,
            "x-guest-name": guestName,
          },
        }
      );

      // Local update
      setMessages((prev) => [...prev, data]);
      setInput("");
    } catch (error) {
      toast.error("Failed to send message");
    }
  };

  return (
    <div className="min-h-screen bg-white dark:bg-black">
      <div className="container mx-auto px-4 py-20">
        <h1 className="mb-8 text-4xl font-bold text-black dark:text-white">
          Peer Communication
        </h1>

        <Card className="mx-auto max-w-4xl">
          <CardContent className="p-0">
            <div className="flex h-[600px] flex-col">
              {/* ✅ Messages Area */}
              <div className="flex-1 space-y-4 overflow-y-auto p-6">
                {messages.length === 0 ? (
                  <div className="flex h-full items-center justify-center">
                    <p className="text-gray-600 dark:text-gray-400">
                      No messages yet. Start the conversation!
                    </p>
                  </div>
                ) : (
                  messages.map((msg, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="flex items-start space-x-3"
                    >
                      {/* Avatar */}
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gray-200 dark:bg-gray-800">
                        <User
                          size={20}
                          className="text-gray-600 dark:text-gray-400"
                        />
                      </div>

                      <div className="flex-1">
                        <div className="mb-1 flex items-center space-x-2">
                          <span className="text-sm font-medium text-black dark:text-white">
                            {msg.guestName || "Guest"}
                          </span>
                          <span className="text-xs text-gray-500 dark:text-gray-500">
                            {new Date(msg.timestamp).toLocaleTimeString()}
                          </span>
                        </div>

                        <p className="rounded-lg bg-gray-100 dark:bg-gray-800 px-4 py-2 text-black dark:text-white">
                          {msg.message}
                        </p>
                      </div>
                    </motion.div>
                  ))
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* ✅ Input Area */}
              <div className="border-t border-gray-200 dark:border-gray-800 p-4">
                <div className="flex items-center space-x-2">
                  <button className="rounded-lg p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800">
                    <Upload size={20} />
                  </button>

                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSend()}
                    placeholder="Type a message..."
                    className="flex-1 rounded-lg border border-gray-300 dark:border-gray-800 bg-white dark:bg-gray-900 px-4 py-2 text-black dark:text-white outline-none focus:border-black dark:focus:border-white"
                  />

                  <Button variant="primary" onClick={handleSend} size="sm">
                    <Send size={18} />
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
