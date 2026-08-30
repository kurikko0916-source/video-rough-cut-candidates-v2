FROM python:3.12-slim AS web
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py index.html app.js styles.css transcript.css ./
COPY config ./config
CMD ["python", "server.py"]

FROM ollama/ollama:latest AS ollama

FROM python:3.12-slim AS worker
ARG WHISPER_CPP_REF=v1.7.6
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    FFMPEG_BIN=/usr/bin/ffmpeg \
    FFPROBE_BIN=/usr/bin/ffprobe \
    WHISPER_BIN=/usr/local/bin/whisper-cli \
    WHISPER_MODEL=/app/models/ggml-large-v3-turbo-q5_0.bin \
    OLLAMA_BIN=/usr/local/bin/ollama \
    OLLAMA_MODELS=/root/.ollama/models \
    OLLAMA_MODEL=qwen3:4b \
    OLLAMA_NO_CLOUD=true
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl git cmake build-essential && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${WHISPER_CPP_REF}" https://github.com/ggml-org/whisper.cpp.git /tmp/whisper.cpp \
    && cmake -S /tmp/whisper.cpp -B /tmp/whisper.cpp/build -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON \
    && cmake --build /tmp/whisper.cpp/build --config Release -j2 \
    && cp /tmp/whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper-cli \
    && rm -rf /tmp/whisper.cpp
COPY --from=ollama /bin/ollama /usr/local/bin/ollama
RUN mkdir -p /app/models /root/.ollama/models \
    && curl -fL --retry 3 https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin -o /app/models/ggml-large-v3-turbo-q5_0.bin
RUN ollama serve >/tmp/ollama.log 2>&1 & pid=$!; sleep 3; ollama pull qwen3:4b; kill $pid
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py index.html app.js styles.css transcript.css ./
COPY config ./config
CMD ["python", "server.py", "--worker"]
