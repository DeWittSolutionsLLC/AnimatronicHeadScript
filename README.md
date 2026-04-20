# Animatronic Head

AI-powered animatronic head using a local LLM (Ollama), Arduino servo control, and offline TTS. Completely free — no API keys, no internet required after setup.

## Hardware

| Component | Arduino Pin |
|-----------|-------------|
| Mouth servo (open/close) | 9 |
| Eyes servo (up/down) | 10 |
| Eyes servo (left/right) | 11 |

Power servo VCC from an **external 5V supply**, not the Arduino 5V pin.

## File structure

```
animatronic-head/
├── arduino/
│   ├── head_control/
│   │   └── head_control.ino   ← upload this to your Arduino
│   └── servo_test.ino         ← use first to calibrate servo angles
├── python/
│   ├── main.py                ← run this to start the head
│   ├── ollama_client.py       ← local LLM via Ollama (streaming)
│   ├── serial_controller.py   ← Arduino serial commands
│   ├── tts_engine.py          ← speech + mouth sync
│   ├── idle_animator.py       ← background idle eye movements
│   └── emotion_map.py         ← emotion → servo angles
└── config/
    └── settings.json          ← all tunable settings
```

## Setup

### 1. Install Ollama
Download from https://ollama.com and install.

```bash
ollama pull llama3.2    # ~2GB download, one time only
ollama serve            # keep this running in a terminal
```

### 2. Install Python dependencies

```bash
pip install pyserial pyttsx3 requests
```

Optional — better TTS with Microsoft neural voices (requires internet):
```bash
pip install edge-tts pygame
```
Then set `tts.engine` to `"edge-tts"` in `settings.json`.

### 3. Calibrate your servos

Upload `arduino/servo_test.ino` to your Arduino.  
Open Serial Monitor at **9600 baud** and send commands like `M60`, `U75`, `L120` to find the right angles for your build.  
Write down the angles, then update `config/settings.json`.

### 4. Upload the main sketch

Upload `arduino/head_control/head_control.ino` to your Arduino.  
Make sure `#define TEST_MODE` is commented out when real servos are connected.

### 5. Configure settings

Edit `config/settings.json`:
- Set `serial.port` to your Arduino's port (`COM3`, `/dev/ttyUSB0`, etc.)
- Update `emotion_servos` with your calibrated angles
- Change `ollama.model` if you pulled a different model

### 6. Run

```bash
cd python
python main.py
```

## Testing without hardware

**Python (no Arduino needed):**  
Set `serial.mock` to `true` in `config/settings.json`. Every servo command prints to the console instead of going to hardware. TTS and mouth animation timing still run normally.

**Arduino (no servos needed):**  
Uncomment `#define TEST_MODE` at the top of `head_control.ino` and upload. The Arduino will process all serial commands and send back normal confirmation strings (`HEAD_READY`, `M:60`, `RESET`) without moving any servos.

## Settings reference (`config/settings.json`)

| Key | Default | Description |
|-----|---------|-------------|
| `serial.port` | `"COM3"` | Arduino serial port |
| `serial.mock` | `false` | `true` to run without Arduino |
| `serial.debug` | `false` | `true` to log every sent/received serial command |
| `ollama.model` | `"llama3.2"` | Ollama model name |
| `ollama.max_history` | `20` | Max messages kept in conversation history |
| `tts.engine` | `"pyttsx3"` | `"pyttsx3"` (offline) or `"edge-tts"` (neural, needs internet) |
| `tts.rate` | `150` | Speech speed in words/min (pyttsx3 only) |
| `tts.voice_index` | `0` | TTS voice index — run `voices` command to list |
| `tts.edge_voice` | `"en-US-GuyNeural"` | Voice name for edge-tts engine |
| `mouth.open_angle` | `60` | Mouth open position |
| `mouth.closed_angle` | `90` | Mouth closed position |
| `mouth.word_open_ms` | `80` | How long mouth stays open per word (ms) |
| `idle.enabled` | `true` | Enable idle eye movements between responses |
| `idle.interval_min` | `2.0` | Min seconds between idle movements |
| `idle.interval_max` | `5.0` | Max seconds between idle movements |
| `idle.jitter` | `12` | Max degrees of random eye offset from neutral |
| `emotion_servos` | — | Servo angles per emotion — tune to your build |

Settings are **hot-reloaded** — save `settings.json` while the head is running and changes apply immediately (no restart needed).

## Runtime commands

Type these at the `You:` prompt:

| Command | Action |
|---------|--------|
| `reset` | Move all servos to neutral |
| `voices` | List available TTS voices |
| `models` | List locally available Ollama models |
| `quit` | Exit |

## Model recommendations

| Model | RAM needed | Notes |
|-------|-----------|-------|
| `phi3:mini` | 4GB | Fast, good for lower-end PCs |
| `llama3.2` | 8GB | Good balance of speed and quality |
| `mistral` | 8GB | Very expressive responses |
| `llama3.1:8b` | 16GB | Best quality |
