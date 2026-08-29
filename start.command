#!/bin/zsh
set -e
cd "${0:A:h}"
python3 server.py &
APP_PID=$!
trap 'kill $APP_PID 2>/dev/null || true' EXIT INT TERM
sleep 1
open http://127.0.0.1:8765
wait $APP_PID
