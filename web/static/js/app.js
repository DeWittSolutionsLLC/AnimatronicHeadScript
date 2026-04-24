/**
 * app.js — Socket.IO chat logic and UI wiring
 */

(function () {
  const socket = io();

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const messages    = document.getElementById("messages");
  const msgInput    = document.getElementById("msg-input");
  const btnSend     = document.getElementById("btn-send");
  const btnLearn    = document.getElementById("btn-learn");
  const btnReset    = document.getElementById("btn-reset");
  const audioPlayer = document.getElementById("audio-player");
  const learningBar = document.getElementById("learning-bar");
  const learningLog = document.getElementById("learning-log");
  const dotLLM      = document.getElementById("dot-llm");
  const dotSerial   = document.getElementById("dot-serial");
  const dotLearning = document.getElementById("dot-learning");

  // ── State ─────────────────────────────────────────────────────────────────
  let isLearning = false;
  let currentUltronMsg = null;  // DOM element being built for streaming

  // ── Helpers ───────────────────────────────────────────────────────────────
  function scrollBottom() {
    messages.scrollTop = messages.scrollHeight;
  }

  function addMessage(role, text, emotion) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    if (role === "ultron" && emotion) {
      const tag = document.createElement("div");
      tag.className = "emotion-tag";
      tag.textContent = emotion;
      div.appendChild(tag);
    }
    const content = document.createElement("span");
    content.textContent = text;
    div.appendChild(content);
    messages.appendChild(div);
    scrollBottom();
    return div;
  }

  function systemMsg(text) {
    addMessage("system", text);
  }

  function setDot(el, state) {
    el.className = "status-dot " + (state || "");
  }

  function setBusy(busy) {
    btnSend.disabled = busy;
    msgInput.disabled = busy;
  }

  // ── Send message ──────────────────────────────────────────────────────────
  function sendMessage() {
    const text = msgInput.value.trim();
    if (!text) return;
    addMessage("user", text);
    msgInput.value = "";
    currentUltronMsg = null;
    setBusy(true);
    socket.emit("message", { text });
  }

  btnSend.addEventListener("click", sendMessage);
  msgInput.addEventListener("keydown", e => {
    if (e.key === "Enter") sendMessage();
  });

  // ── Learning ──────────────────────────────────────────────────────────────
  btnLearn.addEventListener("click", () => {
    if (isLearning) {
      socket.emit("stop_learning");
    } else {
      socket.emit("start_learning");
      learningBar.classList.remove("hidden");
    }
  });

  btnReset.addEventListener("click", () => {
    if (!confirm("Clear conversation and reset servos?")) return;
    socket.emit("reset");
    messages.innerHTML = "";
    systemMsg("Conversation cleared.");
  });

  // ── Socket events ─────────────────────────────────────────────────────────
  socket.on("connect", () => {
    systemMsg("Connected to Ultron.");
  });

  socket.on("disconnect", () => {
    systemMsg("Connection lost.");
    setDot(dotOllama, "error");
  });

  socket.on("status_update", data => {
    if (data.llm_ok !== undefined) setDot(dotLLM, data.llm_ok ? "ok" : "error");
    if (data.serial_ok !== undefined) setDot(dotSerial, data.serial_ok ? "ok" : "error");
    if (data.learning  !== undefined) {
      isLearning = data.learning;
      setDot(dotLearning, data.learning ? "active" : "");
      btnLearn.textContent = data.learning ? "Stop Learning" : "Start Learning";
    }
  });

  // Streaming response: first chunk creates the bubble, rest appends
  socket.on("response_chunk", data => {
    if (!currentUltronMsg) {
      currentUltronMsg = addMessage("ultron", "", data.emotion);
    }
    const content = currentUltronMsg.querySelector("span");
    content.textContent += (content.textContent ? " " : "") + data.text;

    // Pulse neural network nodes for the emotion category
    if (window.neuralPulseCategory) {
      const emotionMap = {
        angry: "quotes", thinking: "quotes", neutral: "traits",
        curious: "references", happy: "song_quotes", sad: "movie_quotes",
        surprised: "movie_quotes",
      };
      const cat = emotionMap[data.emotion] || "quotes";
      window.neuralPulseCategory(cat);
    }
    scrollBottom();
  });

  socket.on("response_done", () => {
    setBusy(false);
    currentUltronMsg = null;
  });

  socket.on("audio_ready", data => {
    audioPlayer.src = data.data;
    audioPlayer.load();
    audioPlayer.play().catch(() => {});
  });

  socket.on("knowledge_update", kb => {
    if (window.neuralUpdate) window.neuralUpdate(kb);
  });

  socket.on("learning_log", data => {
    learningBar.classList.remove("hidden");
    const line = document.createElement("div");
    line.textContent = data.msg;
    learningLog.appendChild(line);
    learningLog.scrollTop = learningLog.scrollHeight;

    // Pulse the whole graph when new knowledge is added
    if (window.neuralPulseCategory) {
      ["quotes", "movie_quotes", "song_quotes", "traits", "references"].forEach(c =>
        window.neuralPulseCategory(c)
      );
    }
  });

  socket.on("learning_status", data => {
    isLearning = data.active;
    setDot(dotLearning, data.active ? "active" : "");
    btnLearn.textContent = data.active ? "Stop Learning" : "Start Learning";
    if (!data.active) learningBar.classList.add("hidden");
  });

  socket.on("chat_cleared", () => {
    messages.innerHTML = "";
  });

  socket.on("error", data => {
    systemMsg("Error: " + (data.msg || "unknown"));
    setBusy(false);
  });
})();
