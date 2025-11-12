import { useEffect, useState, useRef } from "react";
import {
  Mic,
  StopCircle,
  Send,
  Menu,
  X,
  MoreVertical,
  Edit,
  Trash2,
  FolderPlus,
  Search,
  Volume2,
  Play,
  Pause,
} from "lucide-react";
import Button from "../components/ui/Button";
import api from "../lib/api";

const AI_API = ""; // Use Node.js backend which proxies to Flask

export default function Ai() {
  const [history, setHistory] = useState([]);
  const [currentChat, setCurrentChat] = useState(null);
  const [messages, setMessages] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [input, setInput] = useState("");
  const [recording, setRecording] = useState(false);
  const [playingMessageIndex, setPlayingMessageIndex] = useState(null); // Track which message is playing
  const mediaRecorderRef = useRef(null);
  const audioChunks = useRef([]);
  const audioInstancesRef = useRef({}); // Track audio instances by message index

  const chatRef = useRef(null);

  // ✅ Auto scroll down
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  // ✅ Auto-create a chat if none exists
  useEffect(() => {
    if (history.length === 0) {
      newChat();
    } else if (!currentChat) {
      const first = history[0];
      setCurrentChat(first.id);
      setMessages(first.messages);
    }
  }, [history]);

  // ✅ Create new chat session
  const newChat = () => {
    const id = crypto.randomUUID();
    const newChatObj = {
      id,
      name: "New Chat",
      messages: [],
      createdAt: Date.now(),
    };
    setHistory((prev) => [newChatObj, ...prev]);
    setCurrentChat(id);
    setMessages([]);
  };

  // ✅ Generate chat name from first message
  const generateChatName = (text) => {
    return text.split(" ").slice(0, 3).join(" ");
  };

  // ✅ Rename chat
  const renameChat = (id, newName) => {
    setHistory((prev) =>
      prev.map((c) => (c.id === id ? { ...c, name: newName } : c))
    );
  };

  // ✅ Delete chat
  const deleteChat = (id) => {
    const updated = history.filter((c) => c.id !== id);
    setHistory(updated);
    if (updated.length > 0) {
      setCurrentChat(updated[0].id);
      setMessages(updated[0].messages);
    } else {
      newChat();
    }
  };

  // ✅ Send Text Message
  const sendMessage = async () => {
    if (!input.trim()) return;

    let chat = history.find((c) => c.id === currentChat);

    // ✅ Auto-create chat if missing
    if (!chat) {
      newChat();
      chat = history[0];
    }

    const userMessage = { from: "user", text: input };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);

    // ✅ Set chat name based on first message
    if (updatedMessages.length === 1) {
      renameChat(chat.id, generateChatName(input));
    }

    setInput("");

    try {
      const res = await api.post("/api/ai/chat", {
        message: input,
        session_id: chat.id,
      });

      setMessages((prev) => {
        const newMessages = [
          ...prev,
          { 
            from: "bot", 
            text: res.data.response,
            audio: res.data.audio_response, // Store audio for playback
            language: res.data.language_code
          },
        ];
        
        // ✅ Automatically play TTS audio if available (use the new message's index)
        if (res.data.audio_response) {
          setTimeout(() => {
            playAudio(res.data.audio_response, newMessages.length - 1);
          }, 100);
        }
        
        return newMessages;
      });
    } catch (err) {
      console.error("AI Chat Error:", err);
      setMessages((prev) => [
        ...prev,
        { from: "bot", text: "⚠️ AI Server Error. Please check if AI service is running." },
      ]);
    }
  };

  // ✅ File Upload
  const uploadFile = (file) => {
    setMessages((prev) => [
      ...prev,
      { from: "user", text: `📎 Uploaded: ${file.name}` },
    ]);
  };

  const handleFileInput = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "*/*";
    input.onchange = (e) => uploadFile(e.target.files[0]);
    input.click();
  };

  // ✅ Start Voice Recording
  const startRecording = async () => {
    setRecording(true);
    audioChunks.current = [];

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);

    recorder.ondataavailable = (e) => audioChunks.current.push(e.data);
    recorder.onstop = sendVoiceMessage;

    recorder.start();
    mediaRecorderRef.current = recorder;
  };

  // ✅ Stop recording
  const stopRecording = () => {
    setRecording(false);
    mediaRecorderRef.current?.stop();
  };

  // ✅ Convert voice → text → AI response
  const sendVoiceMessage = async () => {
    const audioBlob = new Blob(audioChunks.current, { type: "audio/webm" });
    const formData = new FormData();
    formData.append("audio", audioBlob, "voice.webm");
    
    // Get current chat session ID
    let chat = history.find((c) => c.id === currentChat);
    if (!chat) {
      newChat();
      chat = history[0];
    }
    formData.append("session_id", chat.id);

    try {
      const stt = await api.post("/api/ai/speech-to-text", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const text = stt.data.transcription;
      const detectedLanguage = stt.data.detected_language || stt.data.language_code;

      if (text) {
        // Automatically send the transcribed message with detected language
        const userMessage = { 
          from: "user", 
          text: text,
          language: detectedLanguage // Store the language of the input
        };
        let chat = history.find((c) => c.id === currentChat);
        if (!chat) {
          newChat();
          chat = history[0];
        }
        const updatedMessages = [...messages, userMessage];
        setMessages(updatedMessages);

        // Set chat name based on first message
        if (updatedMessages.length === 1) {
          renameChat(chat.id, generateChatName(text));
        }

        // Send to AI with the detected language to ensure response is in same language
        try {
          const res = await api.post("/api/ai/chat", {
            message: text,
            session_id: chat.id,
            language_code: detectedLanguage, // Pass detected language to ensure response matches
          });

          setMessages((prev) => {
            const newMessages = [
              ...prev,
              { 
                from: "bot", 
                text: res.data.response,
                audio: res.data.audio_response,
                language: res.data.language_code || detectedLanguage // Ensure language is set
              },
            ];
            
            // Automatically play TTS audio if available (use the new message's index)
            if (res.data.audio_response) {
              setTimeout(() => {
                playAudio(res.data.audio_response, newMessages.length - 1);
              }, 100);
            }
            
            return newMessages;
          });
        } catch (err) {
          console.error("AI Chat Error:", err);
          setMessages((prev) => [
            ...prev,
            { from: "bot", text: "⚠️ AI Server Error. Please check if AI service is running." },
          ]);
        }
      }
    } catch (err) {
      console.error("Voice error:", err);
      setMessages((prev) => [
        ...prev,
        { from: "bot", text: "⚠️ Voice recognition failed. Please try typing instead." },
      ]);
    }
  };

  // ✅ Stop all playing audio and reset to start
  const stopAllAudio = () => {
    Object.keys(audioInstancesRef.current).forEach((messageIndex) => {
      const audioData = audioInstancesRef.current[messageIndex];
      if (audioData && audioData.audio) {
        // Reset to start regardless of playing state
        audioData.audio.pause();
        audioData.audio.currentTime = 0;
      }
    });
    setPlayingMessageIndex(null);
  };

  // ✅ Play audio from base64 string for a specific message
  const playAudio = (base64Audio, messageIndex) => {
    try {
      // Stop any currently playing audio from other messages and reset them to start
      stopAllAudio();
      
      // Check if this message already has an audio instance
      let audioData = audioInstancesRef.current[messageIndex];
      
      if (audioData && audioData.audio) {
        // Always reset to start and play from beginning
        audioData.audio.currentTime = 0;
        audioData.audio.play();
        setPlayingMessageIndex(messageIndex);
        return;
      }
      
      // Create new audio instance for this message
      // Decode base64 to binary
      const binaryString = atob(base64Audio);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      // Create blob and play
      const audioBlob = new Blob([bytes], { type: 'audio/webm' });
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      // Set slower playback speed (0.85 = 85% speed, making it slower)
      audio.playbackRate = 0.85;
      
      // Store audio instance for this message
      audioInstancesRef.current[messageIndex] = {
        audio: audio,
        url: audioUrl
      };
      
      // Reset to start
      audio.currentTime = 0;
      
      audio.play().then(() => {
        setPlayingMessageIndex(messageIndex);
      }).catch((err) => {
        console.error("Error playing audio:", err);
        setPlayingMessageIndex(null);
        // Some browsers require user interaction first
        console.log("Audio playback requires user interaction. Click the play button to hear the response.");
      });
      
      // Clean up URL after playback
      audio.onended = () => {
        if (audioInstancesRef.current[messageIndex]) {
          URL.revokeObjectURL(audioUrl);
          delete audioInstancesRef.current[messageIndex];
        }
        setPlayingMessageIndex(null);
      };
      
      // Handle errors
      audio.onerror = () => {
        if (audioInstancesRef.current[messageIndex]) {
          URL.revokeObjectURL(audioUrl);
          delete audioInstancesRef.current[messageIndex];
        }
        setPlayingMessageIndex(null);
        console.error("Error playing audio");
      };
    } catch (err) {
      console.error("Error decoding/playing audio:", err);
      if (audioInstancesRef.current[messageIndex]) {
        delete audioInstancesRef.current[messageIndex];
      }
      setPlayingMessageIndex(null);
    }
  };

  // ✅ Pause audio for a specific message
  const pauseAudio = (messageIndex) => {
    const audioData = audioInstancesRef.current[messageIndex];
    if (audioData && audioData.audio && !audioData.audio.paused) {
      audioData.audio.pause();
      setPlayingMessageIndex(null);
    }
  };

  return (
    <div className="flex h-screen pt-16 bg-white dark:bg-black text-black dark:text-white">

      {/* ✅ Sidebar */}
      <div
        className={`${
          sidebarOpen ? "w-64" : "w-0"
        } transition-all bg-gray-100 dark:bg-gray-900 border-r border-gray-300 dark:border-gray-700 overflow-hidden`}
      >
        <div className="p-4 border-b border-gray-300 dark:border-gray-700 flex items-center justify-between">
          <h2 className="font-bold text-lg">Chats</h2>
          <Button size="sm" onClick={newChat}>
            New Chat
          </Button>
        </div>

        {/* Search */}
        <div className="p-3 flex items-center gap-2 border-b border-gray-300 dark:border-gray-700">
          <Search size={18} />
          <input
            className="bg-transparent w-full outline-none"
            placeholder="Search chats"
          />
        </div>

        {/* Chat History */}
        <div className="overflow-y-auto h-full">
          {history.map((chat) => (
            <div
              key={chat.id}
              className={`px-4 py-3 border-b border-gray-300 dark:border-gray-800 cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 ${
                currentChat === chat.id ? "bg-gray-300 dark:bg-gray-700" : ""
              }`}
              onClick={() => {
                setCurrentChat(chat.id);
                setMessages(chat.messages);
              }}
            >
              <div className="flex justify-between">
                <span className="truncate">{chat.name}</span>

                {/* 3 dots menu */}
                <div className="relative group">
                  <MoreVertical size={18} />

                  <div className="hidden group-hover:block absolute right-0 mt-2 bg-gray-800 text-white p-2 rounded shadow-lg z-50 w-32">
                    <button
                      className="flex items-center gap-2 w-full text-left p-1 hover:bg-gray-700"
                      onClick={() => {
                        const name = prompt("Rename chat:", chat.name);
                        if (name) renameChat(chat.id, name);
                      }}
                    >
                      <Edit size={14} /> Edit
                    </button>

                    <button
                      className="flex items-center gap-2 w-full text-left p-1 hover:bg-gray-700"
                      onClick={() => deleteChat(chat.id)}
                    >
                      <Trash2 size={14} /> Delete
                    </button>
                  </div>
                </div>
              </div>

              <hr className="border-gray-400/40 mt-2" />
            </div>
          ))}
        </div>
      </div>

      {/* ✅ Toggle Button */}
      <button
        className="absolute left-2 top-20 bg-gray-200 dark:bg-gray-800 px-2 py-1 rounded"
        onClick={() => setSidebarOpen(!sidebarOpen)}
      >
        {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* ✅ Chat Window */}
      <div className="flex-1 flex flex-col">
        <div
          ref={chatRef}
          className="flex-1 overflow-y-auto p-6 space-y-4"
        >
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`max-w-xl px-4 py-2 rounded-xl flex items-start gap-2 ${
                msg.from === "user"
                  ? "ml-auto bg-blue-600 text-white"
                  : "bg-gray-200 dark:bg-gray-800"
              }`}
            >
              <div className="flex-1">{msg.text}</div>
              {msg.from === "bot" && msg.audio && (
                <div className="flex items-center gap-1">
                  {playingMessageIndex === i ? (
                    <button
                      onClick={() => pauseAudio(i)}
                      className="p-1.5 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-700 transition-colors"
                      title="Pause audio"
                    >
                      <Pause size={18} className="text-blue-600 dark:text-blue-400" />
                    </button>
                  ) : (
                    <button
                      onClick={() => playAudio(msg.audio, i)}
                      className="p-1.5 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-700 transition-colors"
                      title="Play audio response"
                    >
                      <Volume2 size={18} className="text-blue-600 dark:text-blue-400" />
                    </button>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* ✅ Input Section */}
        <div className="p-4 border-t border-gray-300 dark:border-gray-700 flex items-center gap-2">
          {/* File Upload */}
          <button
            onClick={handleFileInput}
            className="p-2 rounded-lg bg-gray-200 dark:bg-gray-800"
          >
            <FolderPlus size={22} />
          </button>

          {/* Voice */}
          {recording ? (
            <button
              onClick={stopRecording}
              className="p-2 bg-red-500 text-white rounded-lg"
            >
              <StopCircle size={24} />
            </button>
          ) : (
            <button
              onClick={startRecording}
              className="p-2 bg-gray-200 dark:bg-gray-800 rounded-lg"
            >
              <Mic size={22} />
            </button>
          )}

          <input
            className="flex-1 px-3 py-2 rounded-lg bg-gray-200 dark:bg-gray-900 outline-none"
            placeholder="Ask something..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />

          <button
            className="p-3 bg-blue-600 text-white rounded-lg"
            onClick={sendMessage}
          >
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
}
