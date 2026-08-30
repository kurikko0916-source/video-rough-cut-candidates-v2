#!/bin/zsh
set -e
cd "${0:A:h}"
RUNTIME_DIR="$PWD/tools/runtime"
export PATH="$RUNTIME_DIR/bin:$PATH"
export FFMPEG_BIN="$RUNTIME_DIR/bin/ffmpeg"
export FFPROBE_BIN="$RUNTIME_DIR/bin/ffprobe"
export CLOUD_BRIDGE_URL="https://video-rough-cut-ai-695115803909.asia-northeast1.run.app"
export PORT=8766

"$RUNTIME_DIR/bin/python3" server.py &
APP_PID=$!
trap 'kill $APP_PID 2>/dev/null || true' EXIT INT TERM
sleep 1
open http://127.0.0.1:8766
wait $APP_PID
