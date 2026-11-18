# Voice Chat — how it works and how to use it

## 1. The big picture

Everything happens on your own PC. Nothing is sent to the internet.

```
 you speak ──► 1. MICROPHONE ──► 2. SPEECH → TEXT ──► 3. LOCAL AI REPLY ──► 4. TEXT → SPEECH ──► you hear it
               (record + find      (Whisper models      (Ollama +              (Kokoro for English,
                where speech is)    on the GPU)          Gemma 4 12B)           Bangla VITS voice)
```

Each step is a separate piece of software. The program `server.py` connects them, and the web page
(`web/index.html`) is what you click on.

---

## 2. Each piece, explained simply

### Step 1 — Microphone and "is someone talking?" (VAD)

| What | Details |
|---|---|
| Browser page | Records your voice in the browser as raw sound (no compression), then sends it to the server as a WAV file. |
| Hands-free mode | Uses the Python library **sounddevice** to read the microphone directly. |
| Silero VAD | A tiny AI model (2 MB) that answers one question 30 times a second: *is this speech or silence?* It finds where your sentence starts and ends, and cuts out the silence. Silence matters because Whisper tends to invent words when it hears nothing. |
| Files | `voice/mic.py`, `voice/vad.py` |

### Step 2 — Speech to text (STT)

| What | Details |
|---|---|
| **Whisper** | OpenAI's open speech-recognition model. It listens to audio and writes down the words. |
| **faster-whisper** | A fast engine for running Whisper on your graphics card (it uses a library called CTranslate2). |
| English model | **Whisper large-v3** — the biggest general Whisper model. Measured: 4.5 % of words wrong on 100 test clips. |
| Bangla model | **tugstugi Bengali.AI Whisper-medium** — the model from the Kaggle notebook you gave me. It won the Bengali.AI speech competition. I downloaded it from Hugging Face and **converted** it to the faster-whisper format (`tools/convert_model.py`). Measured: 16.4 % of words wrong (4.9 % of letters), on public test speech. |
| Long recordings | Cut at your pauses into pieces of at most 24 seconds, each written down separately. |
| Files | `voice/stt.py`, models in `models/` |

**Which language did you speak?** (only when the language box says *Auto*)
Three clues are combined (`voice/lid.py`):

1. Whisper's own guess of the language.
2. **VoxLingua107** — a separate small model (SpeechBrain library, runs on the CPU in ~0.07 s) trained only to recognise languages. It is excellent at spotting Bangla, but it often mistakes Bangladeshi/Indian-accented English for Bangla.
3. How cleanly the audio can be written down as English. Real English (any accent) writes down cleanly; Bangla forced into English does not.

A small formula, fitted on 901 test recordings, weighs the three clues. Errors went from 32 to 9 out of 901, and
the remaining ones are almost all 2-second phrases that start with an English name.

**If it still picks the wrong language:** under your message, click **"wrong language? hear it as …"**. Your same
recording is written down again in the other language, the wrong reply disappears, and a new reply is given.
If you choose **English** or **বাংলা** in the top bar instead of Auto, this step is skipped and that language is
always used — that is the most reliable choice when you know what you will speak.

### Customer-care mode (the default)

The AI plays **Mitu, a customer-care agent** for a company. Everything she knows about the company is in
**`business/profile.md`**: products, delivery charges and times, returns, refunds, warranty, offers, and what she
may and may not promise. The shipped profile is a *fictional* shop (Nirvana Gadgets) — **replace it with your
real business** and restart the app. Change her name, the company name and the greeting in `config.json`:

```json
{ "agent_name": "Rina", "company_name": "My Shop", "greeting": "আসসালামু আলাইকুম, মাই শপ থেকে রিনা বলছি।" }
```

How she is told to behave is in `voice/prompts.py`: talk like a person on the phone (one or two short sentences,
one question at a time), natural Bangladeshi Bangla instead of textbook Bangla, acknowledge feelings only when
the caller is upset, use the rules in the caller's favour, never invent stock or order status (she notes it for
the team, who call back within 24 hours), never end on a bare "no", and offer a senior colleague's call-back to
angry callers. Set `"mode": "assistant"` in `config.json` for the old general-purpose helper.

How it was measured: 24 realistic calls (`business/scenarios.json`, 12 English + 12 Bangla: late delivery,
wrong item, angry repeat caller, discount pushing, unknown stock…) were scored by independent reviewers on
human-likeness, empathy, brevity, correctness and language. From the old setup to the final one: Bangla
human-likeness 1.5 → 3.6 out of 5, correctness 1.3 → 4.0; English human-likeness 1.9 → 3.6, correctness
1.6 → 4.5. Replies now start in about 0.2 s (thinking is switched off; it used to add 6–14 s of silence).
Re-run with `tools\care_eval.py <name>` after you change the profile or the prompt.

### Step 3 — The AI reply (LLM)

| What | Details |
|---|---|
| **Ollama** | A free program that runs AI chat models locally and offers them through a small web address on your PC (`http://127.0.0.1:11434`). It was installed to `%LOCALAPPDATA%\Programs\Ollama`. |
| **gemma4:12b-it-qat** | Google's Gemma 4 model, 12 billion parameters, compressed to 7 GB so it fits on your 16 GB graphics card. It writes good Bangla and English. Speed: about 50 words-pieces (tokens) per second. |
| Instructions to the model | "Reply in the same language the user used; write plain text." Your last 20 exchanges are remembered until you click **New chat**. |
| Files | `voice/llm.py` |

### Step 4 — Speaking the reply (TTS = text to speech)

| What | Details |
|---|---|
| **sherpa-onnx** | A free engine that runs speech voices on the CPU, offline. Version **1.12.40** is required (version 2 skips words it does not know). |
| English voice | **Kokoro-82M**, voice "af_heart" — one of the most natural small open voices. A sentence is ready in about 1.2 s. |
| Bangla voice | **Coqui VITS Bangla female** — a voice trained on Bangla. A sentence is ready in about 0.7 s. |
| Smart reading | Numbers in Bangla replies are read as Bangla words (2024 → দুই হাজার চব্বিশ). English words inside a Bangla sentence ("laptop", "RAM") are read by the English voice. |
| Sentence by sentence | Speaking starts as soon as the first sentence of the reply is finished, while the rest is still being written. |
| Files | `voice/tts.py`, `voice/bn_numbers.py`, voices in `models/tts/` |

### The web app

| What | Details |
|---|---|
| **FastAPI + Uvicorn** | The Python web server in `server.py`, at `http://127.0.0.1:8765`. |
| Page | `web/index.html` — microphone button, language box, Voice on/off, New chat. |
| Server addresses (API) | `/api/talk` (audio → text → reply), `/api/transcribe` (audio → text only), `/api/chat` (typed text → reply), `/api/speak` (text → WAV sound), `/api/status`. |

### How the graphics card memory is used (16 GB)

| Part | Memory |
|---|---|
| Whisper large-v3 (English) | about 3 GB |
| Bangla Whisper-medium | about 1.5 GB |
| Gemma 4 12B reply model | about 8 GB |
| Total | about 14.6 GB — so close other big GPU programs (games, Xyrus) while using it |

---

## 3. Folder map

```
D:\voice
├── start.bat              ← double-click: starts everything and opens the page
├── talk.bat               ← hands-free mode in a terminal window
├── server.py              ← the web server that connects all the steps
├── talk.py                ← the hands-free program
├── config.json            ← (optional) your own settings, see section 5
├── requirements.txt       ← the exact Python package versions that work
├── web\index.html         ← the page you see
├── voice\                 ← the building blocks
│   ├── stt.py  lid.py  vad.py  mic.py  llm.py  tts.py  bn_numbers.py  config.py  runtime.py
├── models\                ← the AI models (about 8 GB)
│   ├── whisper-large-v3\  bn-tugstugi-medium\  bn-mozilla-large-v3\  lid-voxlingua107\  tts\
├── tools\                 ← benchmark, conversion and setup scripts
├── logs\                  ← test results
└── .venv\                 ← the private Python installation for this project
```

---

## 4. How to run it

### Easiest: the desktop shortcut

Double-click **Voice Chat** on the desktop. Wait about 15 seconds; the page opens by itself.

- Click the microphone (or hold **Space**), speak, click again (or release Space).
- Allow the microphone the first time the browser asks.
- Close the black window to stop everything.

### From a terminal

Start the web app:

```bash
D:/voice/.venv/Scripts/python.exe D:/voice/server.py
```

Then open http://127.0.0.1:8765 in Chrome or Edge.

Hands-free mode (listens all the time, answers out loud). Close the web app first — both together do not fit on the graphics card:

```bash
D:/voice/.venv/Scripts/python.exe D:/voice/talk.py
```

Hands-free, always Bangla:

```bash
D:/voice/.venv/Scripts/python.exe D:/voice/talk.py --lang bn
```

Hands-free, always English:

```bash
D:/voice/.venv/Scripts/python.exe D:/voice/talk.py --lang en
```

Dictation only — speech to text, no AI reply:

```bash
D:/voice/.venv/Scripts/python.exe D:/voice/talk.py --no-llm
```

Reply as text without speaking it:

```bash
D:/voice/.venv/Scripts/python.exe D:/voice/talk.py --quiet
```

Write down a recording you already have (any audio file):

```bash
D:/voice/.venv/Scripts/python.exe D:/voice/talk.py --no-llm --file D:/path/to/recording.wav
```

List your microphones, then choose one by its number:

```bash
D:/voice/.venv/Scripts/python.exe D:/voice/talk.py --devices
```

```bash
D:/voice/.venv/Scripts/python.exe D:/voice/talk.py --device 1
```

### Ollama commands (the AI reply model)

See which models are installed:

```bash
ollama list
```

See which model is loaded right now and whether it runs on the graphics card ("100% GPU"):

```bash
ollama ps
```

Free its graphics-card memory when you are done:

```bash
ollama stop gemma4:12b-it-qat
```

Chat with the model directly, without voice:

```bash
ollama run gemma4:12b-it-qat
```

(If `ollama` is "not recognized", open a new terminal, or use `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`.)

### Measure accuracy yourself

English (100 public test recordings):

```bash
D:/voice/.venv/Scripts/python.exe D:/voice/tools/bench.py --lang en large-v3=models/whisper-large-v3
```

Bangla:

```bash
D:/voice/.venv/Scripts/python.exe D:/voice/tools/bench.py --lang bn tugstugi=models/bn-tugstugi-medium
```

Language detection (Auto mode) on 901 recordings:

```bash
D:/voice/.venv/Scripts/python.exe D:/voice/tools/lang_check.py
```

---

## 5. Changing settings

Create a file `D:\voice\config.json`. Only write the settings you want to change. Example:

```json
{
  "speak_replies": true,
  "tts_english_speed": 1.1,
  "tts_bangla_speed": 0.95,
  "llm_model": "gemma4:12b-it-qat",
  "end_silence_s": 1.5
}
```

| Setting | What it does |
|---|---|
| `speak_replies` | `false` = never speak replies |
| `tts_english_speed`, `tts_bangla_speed` | 1.0 normal, bigger = faster |
| `tts_english_voice` | Kokoro voice number (3 = af_heart; 0–52 are other voices) |
| `llm_model` | any model you have in `ollama list` |
| `end_silence_s` | hands-free: seconds of silence that end your sentence (bigger if it cuts you off) |
| `bangla_model` | e.g. `"models/bn-mozilla-large-v3"` to try the other Bangla model |
| `voxlingua` | `false` = use Whisper alone to detect the language |

Restart the app after changing it.

---

## 6. If something goes wrong

| Problem | Fix |
|---|---|
| "Microphone not available" | Use the page in Chrome/Edge (not the Claude app's built-in browser), click the padlock left of the address, allow Microphone, reload. |
| "replies off: Ollama is not running" | Start Ollama from the Start menu, or run `ollama serve`. |
| Wrong language detected | Click "wrong language? hear it as …" under your message, or pick English / বাংলা in the top bar instead of Auto. |
| It writes wrong words | Speak 10–20 cm from the mic, in a quiet room, and finish the sentence before stopping. |
| No sound from replies | Check the 🔊 Voice on button, and that Windows' output device is your speakers and not muted. |
| Very slow replies | Run `ollama ps`; it must say 100% GPU. Close games or other GPU programs. |
| Hands-free says the app is already running | Close the Voice Chat window first (only one can use the graphics card). |

---

## 7. Honest limits

- No speech recogniser is 100 % accurate. Measured on public test speech: English 4.5 % of words wrong;
  Bangla 14.8 % of words and 3.5 % of letters (most "word" errors are spelling or spacing variants such as
  "অপ্রাপ্ত বয়স্ক" vs "অপ্রাপ্তবয়স্ক"). Searching harder (beam 10) gave no improvement. Strong regional
  dialects are harder. The biggest remaining lever would be training the Bangla model on your own voice.
- Auto language detection is wrong about 1 time in 100, most often with short phrases or strongly accented English.
- The AI can be wrong about facts and has no internet, so it cannot know today's weather or news.
