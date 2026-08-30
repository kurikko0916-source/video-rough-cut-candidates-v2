#!/bin/zsh
set -e
cd "${0:A:h}"
RUNTIME_DIR="$PWD/tools/runtime"
OLLAMA_BIN="$PWD/tools/Ollama.app/Contents/Resources/ollama"
export PATH="$RUNTIME_DIR/bin:$PATH"
export FFMPEG_BIN="$RUNTIME_DIR/bin/ffmpeg"
export FFPROBE_BIN="$RUNTIME_DIR/bin/ffprobe"
export WHISPER_BIN="$RUNTIME_DIR/bin/whisper-cli"
export WHISPER_MODEL="$PWD/models/ggml-large-v3-turbo-q5_0.bin"
export WHISPER_VAD_BIN="$RUNTIME_DIR/bin/whisper-vad-speech-segments"
export WHISPER_VAD_MODEL="$PWD/models/ggml-silero-v6.2.0.bin"
export OLLAMA_BIN
export OLLAMA_MODELS="$PWD/tools/ollama-models"
export OLLAMA_MODEL="qwen3:4b"
export OLLAMA_NO_CLOUD=true

"$RUNTIME_DIR/bin/python3" server.py &
APP_PID=$!
trap 'kill $APP_PID 2>/dev/null || true' EXIT INT TERM
sleep 1
open http://127.0.0.1:8765
wait $APP_PID
