# Voice Chat

Speak in English or Bangla → your words appear as text → a local LLM replies. Everything runs on this PC.

## Use it

- **Desktop shortcut "Voice Chat"** (or `start.bat`): starts everything and opens http://127.0.0.1:8765.
  Click the microphone (or hold Space), speak, click again. Pick *English* or *বাংলা* in the top bar when
  you know which language you will speak — that is more accurate than *Auto*.
- **"Voice Chat (hands-free)"** (or `talk.bat`): a terminal window that listens continuously and answers
  each time you stop talking. Options: `talk.bat --lang bn`, `--no-llm` (dictation only),
  `--file clip.wav`, `--devices`.

Use the page in your normal browser (Chrome/Edge) and allow the microphone when asked.

## Install

Needs Windows, Python 3.13, an NVIDIA GPU with 16 GB (tested on an RTX 5060 Ti) and about 15 GB of disk.
The models are not in this repository (several GB); the steps below download them into `models/`.

1. Python environment:

       py -3.13 -m venv .venv
       .venv\Scripts\pip install -r requirements.txt

2. Speech-to-text and language-ID models (from Hugging Face):

       .venv\Scripts\hf download Systran/faster-whisper-large-v3 --local-dir models\whisper-large-v3
       .venv\Scripts\hf download speechbrain/lang-id-voxlingua107-ecapa --local-dir models\lid-voxlingua107

   The Bangla model is converted from `bengaliAI/tugstugi_bengaliai-asr_whisper-medium` with
   `tools/convert_model.py` (run it with a Python that has `torch`, `transformers` and `ctranslate2`):

       python tools\convert_model.py bengaliAI/tugstugi_bengaliai-asr_whisper-medium models\bn-tugstugi-medium

3. Voices (from the sherpa-onnx `tts-models` release), unpacked into `models\tts`:
   `kokoro-multi-lang-v1_0.tar.bz2` and `vits-coqui-bn-custom_female.tar.bz2` from
   https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

4. Reply model: install [Ollama](https://ollama.com), then

       ollama pull gemma4:12b-it-qat

## How it works

| Step | What runs |
|---|---|
| Speech → text, English | Whisper large-v3 (`models/whisper-large-v3`) |
| Speech → text, Bangla | tugstugi's Bengali.AI Whisper-medium (`models/bn-tugstugi-medium`), falling back to `models/bn-mozilla-large-v3` until it is installed |
| Which language? (Auto) | three signals combined (`voice/lid.py`): Whisper large-v3's language guess, the VoxLingua107 language-ID model (`models/lid-voxlingua107`, CPU), and how cleanly the audio transcribes as English. On 901 test clips it chose wrong 8 times vs 32 for Whisper alone; South Asian-accented English is the hard case |
| Long recordings | cut at pauses into pieces of at most 24 s, each decoded on its own |
| Reply | Ollama with `gemma4:12b-it-qat`, streamed |
| Speaking the reply, English | Kokoro-82M (`models/tts/kokoro-multi-lang-v1_0`, voice af_heart), offline on the CPU |
| Speaking the reply, Bangla | Coqui VITS Bangla female voice (`models/tts/vits-coqui-bn-custom_female`), offline on the CPU |

Replies are spoken sentence by sentence while they are still being written, so the first words start
about a second after the first sentence is complete. The **🔊 Voice on/off** button in the top bar turns
it off; it remembers your choice. Speaking stops as soon as you press the microphone. In hands-free mode
the microphone is ignored while the reply is being spoken (so it does not answer itself); `talk.bat --quiet`
prints replies without speaking them. Set `"speak_replies": false` in `config.json` to turn speaking off
everywhere.

Both speech models run on the GPU through faster-whisper. Settings live in `voice/config.py`; override any
of them in a `config.json` in this folder, e.g. `{"llm_model": "qwen3.5:9b", "bangla_model": "models/bn-mozilla-large-v3"}`.

## Accuracy

No speech recogniser is 100 % accurate. Measure this setup on public test speech (FLEURS, 100 clips):

    .venv\Scripts\python.exe tools\bench.py --lang en large-v3=models/whisper-large-v3
    .venv\Scripts\python.exe tools\bench.py --lang bn tugstugi=models/bn-tugstugi-medium

Results are kept in `logs/bench_*.json`. What helps most: choose the language instead of Auto, keep the
microphone 10–20 cm from your mouth, a quiet room, and finish the sentence before stopping.

## Files

- `server.py`, `web/index.html` — the browser app (API: `/api/talk`, `/api/transcribe`, `/api/chat`)
- `talk.py` — hands-free terminal mode
- `voice/stt.py` — transcription; `voice/mic.py`, `voice/vad.py` — microphone + speech detection; `voice/llm.py` — replies
- `voice/tts.py` — speaking (API: `/api/speak`); `requirements.txt` — the tested package versions (keep the pins)
- `tools/finish_setup.ps1` — finishes installation after the downloads (Ollama, reply model, Bangla model)
- `tools/convert_model.py` — converts a Hugging Face Whisper fine-tune for faster-whisper
- `tools/lang_eval.py`, `tools/lid_voxlingua.py`, `tools/lid_fit.py`, `tools/lang_check.py` — how the language rule was measured and fitted
- `tools/bench.py` — accuracy benchmark; `tools/fetch.py` — parallel resumable downloader
