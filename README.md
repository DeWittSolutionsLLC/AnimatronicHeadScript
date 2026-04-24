# Animatronic Head

AI-powered animatronic head using Google Gemini, Arduino servo control, and offline TTS.

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
│   ├── llm_client.py          ← Google Gemini API client (streaming)
│   ├── serial_controller.py   ← Arduino serial commands
│   ├── tts_engine.py          ← speech + mouth sync
│   ├── idle_animator.py       ← background idle eye movements
│   └── emotion_map.py         ← emotion → servo angles
└── config/
    └── settings.json          ← all tunable settings
```

## Setup

### 1. Get a Gemini API key
Go to https://aistudio.google.com/apikey, create a free API key, and either:
- Set the environment variable: `set GEMINI_API_KEY=your_key_here`
- Or paste it into `config/settings.json` under `gemini.api_key`

### 2. Install Python dependencies

```bash
pip install pyserial pyttsx3 google-generativeai
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
- Change `gemini.model` if you want a different Gemini model (e.g. `gemini-1.5-pro`)

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
| `gemini.api_key` | `""` | Gemini API key (or use `GEMINI_API_KEY` env var) |
| `gemini.model` | `"gemini-2.0-flash"` | Gemini model name |
| `gemini.max_history` | `20` | Max messages kept in conversation history |
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
| `models` | List available Gemini models |
| `quit` | Exit |

## Model recommendations

| Model | Notes |
|-------|-------|
| `gemini-2.0-flash` | Default — fast, low cost, good quality |
| `gemini-1.5-pro` | Higher quality, larger context window |
| `gemini-1.5-flash` | Budget option with solid performance |
