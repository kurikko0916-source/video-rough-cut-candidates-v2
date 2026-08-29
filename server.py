#!/usr/bin/env python3
"""粗カットAI - 外部APIを使わないローカルHTTPサーバー。"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
JOBS_DIR = ROOT / "data" / "jobs"
MODELS_DIR = ROOT / "models"
HOST, PORT = "127.0.0.1", 8765
MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def tool_path(env_name: str, default_name: str) -> str | None:
    configured = os.getenv(env_name)
    if configured and Path(configured).exists(): return configured
    local_candidates = {
        "ffmpeg": ROOT / "tools" / "runtime" / "bin" / "ffmpeg",
        "ffprobe": ROOT / "tools" / "runtime" / "bin" / "ffprobe",
        "whisper-cli": ROOT / "tools" / "runtime" / "bin" / "whisper-cli",
        "ollama": ROOT / "tools" / "Ollama.app" / "Contents" / "Resources" / "ollama",
    }
    local = local_candidates.get(default_name)
    return str(local) if local and local.exists() else shutil.which(default_name)


def whisper_model() -> Path:
    configured = os.getenv("WHISPER_MODEL")
    return Path(configured).expanduser() if configured else MODELS_DIR / "ggml-large-v3-turbo-q5_0.bin"


def environment() -> dict:
    ffmpeg = tool_path("FFMPEG_BIN", "ffmpeg")
    ffprobe = tool_path("FFPROBE_BIN", "ffprobe")
    whisper = tool_path("WHISPER_BIN", "whisper-cli")
    ollama = tool_path("OLLAMA_BIN", "ollama")
    model = whisper_model()
    return {"tools": {
        "ffmpeg": {"ready": bool(ffmpeg), "path": ffmpeg}, "ffprobe": {"ready": bool(ffprobe), "path": ffprobe},
        "whisper": {"ready": bool(whisper), "path": whisper}, "whisper_model": {"ready": model.is_file(), "path": str(model)},
        "ollama": {"ready": bool(ollama), "path": ollama},
    }}


def update(job_id: str, **values) -> None:
    with LOCK:
        JOBS[job_id].update(values)
        snapshot = dict(JOBS[job_id])
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "status.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def run(command: list[str], error_message: str) -> subprocess.CompletedProcess:
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if process.returncode:
        detail = (process.stderr or process.stdout).strip()[-1200:]
        raise RuntimeError(f"{error_message}\n{detail}")
    return process


def probe_video(video: Path, ffprobe: str) -> dict:
    output = run([ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type,r_frame_rate", "-of", "json", str(video)], "動画情報を読み取れませんでした。")
    data = json.loads(output.stdout)
    duration = float(data.get("format", {}).get("duration") or 0)
    fps = 0.0
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            numerator, _, denominator = str(stream.get("r_frame_rate", "0/1")).partition("/")
            fps = float(numerator) / float(denominator or 1)
            break
    return {"duration_seconds": duration, "source_fps": fps}


SILENCE_RE = re.compile(r"silence_(start|end):\s*([0-9.]+)")


def extract_audio_and_silence(video: Path, audio: Path, ffmpeg: str) -> list[dict]:
    command = [ffmpeg, "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-af", "silencedetect=noise=-38dB:d=1.8", "-c:a", "pcm_s16le", str(audio)]
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if process.returncode:
        raise RuntimeError("動画から音声を取り出せませんでした。\n" + process.stderr[-1200:])
    silences, start = [], None
    for kind, value in SILENCE_RE.findall(process.stderr):
        if kind == "start": start = float(value)
        elif start is not None:
            end = float(value)
            if end - start >= 2.2: silences.append({"start": start, "end": end})
            start = None
    return silences


def timestamp_seconds(value) -> float:
    if isinstance(value, (int, float)): return float(value) / 1000 if value > 100000 else float(value)
    text = str(value or "0").replace(",", ".")
    parts = text.split(":")
    try:
        if len(parts) == 3: return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return float(text)
    except ValueError: return 0.0


def parse_whisper_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    source = data.get("transcription") or data.get("segments") or []
    segments = []
    for item in source:
        offsets = item.get("offsets") or {}
        timestamps = item.get("timestamps") or {}
        start = offsets.get("from", item.get("start", timestamps.get("from", 0)))
        end = offsets.get("to", item.get("end", timestamps.get("to", 0)))
        if offsets: start, end = float(start) / 1000, float(end) / 1000
        else: start, end = timestamp_seconds(start), timestamp_seconds(end)
        text = str(item.get("text") or "").strip()
        if text: segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    if not segments: raise RuntimeError("文字起こし結果から発言を取得できませんでした。")
    return segments


def transcribe(audio: Path, job_dir: Path, whisper: str, model: Path) -> list[dict]:
    prefix = job_dir / "transcript"
    run([whisper, "-m", str(model), "-f", str(audio), "-l", "ja", "-ojf", "-of", str(prefix), "-np"], "日本語の文字起こしに失敗しました。")
    result = prefix.with_suffix(".json")
    if not result.exists(): raise RuntimeError("文字起こし結果が作成されませんでした。")
    return parse_whisper_json(result)


RETAKE = re.compile(r"(もう一回|もう一度|今のなし|今のはなし|改めて|最初から|言い直し|撮り直し|ここはカット|カットで|どこからでした|次どこから)")
FILLER_ONLY = re.compile(r"^[、。\s]*(あの+|え+と|え+|う+|まあ+|その+|なんていうか)[、。\sー…]*$")


def resume_text(segments: list[dict], end: float) -> str:
    for segment in segments:
        if segment["start"] >= end - .12: return segment["text"][:100]
    return ""


def rule_candidates(segments: list[dict], silences: list[dict]) -> list[dict]:
    candidates = []
    for index, segment in enumerate(segments):
        duration = segment["end"] - segment["start"]
        if RETAKE.search(segment["text"]):
            next_index = min(index + 1, len(segments) - 1)
            end = segments[next_index]["start"] if next_index > index else segment["end"]
            candidates.append({"start_seconds": segment["start"], "end_seconds": end, "resume_text": segments[next_index]["text"] if next_index > index else "", "category": "仕切り直し", "confidence": "high", "source": "rule"})
        elif duration >= 3.0 and FILLER_ONLY.match(segment["text"]):
            candidates.append({"start_seconds": segment["start"], "end_seconds": segment["end"], "resume_text": resume_text(segments, segment["end"]), "category": "長い言い淀み", "confidence": "high", "source": "rule"})
    for silence in silences:
        start, end = silence["start"], silence["end"]
        if end - start >= 2.8:
            candidates.append({"start_seconds": start + .25, "end_seconds": max(start + .25, end - .18), "resume_text": resume_text(segments, end), "category": "長い無音", "confidence": "high", "source": "silence"})
    return candidates


def ollama_available(model: str) -> bool:
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as response:
            names = [item.get("name", "") for item in json.load(response).get("models", [])]
        return model in names or any(name.split(":")[0] == model.split(":")[0] for name in names)
    except (URLError, TimeoutError, OSError, ValueError): return False


def llm_candidates(segments: list[dict], model: str) -> list[dict]:
    found = []
    for offset in range(0, len(segments), 55):
        chunk = segments[max(0, offset - 3): offset + 58]
        transcript = "\n".join(f'[{s["start"]:.2f}-{s["end"]:.2f}] {s["text"]}' for s in chunk)
        prompt = f"""/no_think
あなたは建築・住宅YouTubeの保守的な粗カット補助です。次の日本語対談から、高い確信度で不要なまとまった区間だけを選びます。
対象: 撮影中の打ち合わせ、仕切り直し、失敗テイク、直後に完全に言い直した前半、追加情報のない明らかな重複、長く内容のない言い淀み。
対象外: 短い「あのー」「えーと」単体、雑談、専門説明、長いが内容のある説明、追加情報のある繰り返し。迷う場合は必ず残します。
判定時刻は提供した発言境界から選び、JSONのみを返してください。
{{"candidates":[{{"start_seconds":0.0,"end_seconds":0.0,"resume_text":"削除後の最初の発言","category":"種類","confidence":"high"}}]}}

{transcript}"""
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "think": False, "format": "json", "options": {"temperature": 0.1}}, ensure_ascii=False).encode("utf-8")
        try:
            request = Request("http://127.0.0.1:11434/api/chat", data=body, headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=300) as response: content = json.load(response)["message"]["content"]
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            json_block = re.search(r"\{.*\}", content, flags=re.DOTALL)
            parsed = json.loads(json_block.group(0) if json_block else content)
            for item in parsed.get("candidates", []):
                if item.get("confidence") == "high": item["source"] = "local_llm"; found.append(item)
        except (URLError, TimeoutError, OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return found


def snap_and_merge(candidates: list[dict], segments: list[dict], duration: float) -> list[dict]:
    valid = []
    boundaries = sorted({0.0, duration, *[s["start"] for s in segments], *[s["end"] for s in segments]})
    def nearest(value): return min(boundaries, key=lambda point: abs(point - value)) if boundaries else value
    for item in candidates:
        try: start, end = float(item["start_seconds"]), float(item["end_seconds"])
        except (KeyError, TypeError, ValueError): continue
        is_llm = item.get("source") == "local_llm"
        if is_llm: start, end = nearest(start), nearest(end)
        start, end = max(0, start), min(duration, end)
        if end - start < .7 or end <= start: continue
        category = str(item.get("category") or "AI判定")
        if is_llm and category not in {"撮影中の打ち合わせ", "仕切り直し", "失敗テイク", "言い直し", "明らかな重複", "長い言い淀み"}: category = "AI判定"
        resume = resume_text(segments, end) if is_llm else str(item.get("resume_text") or resume_text(segments, end))
        valid.append({"start_seconds": round(start, 3), "end_seconds": round(end, 3), "resume_text": resume[:120], "category": category, "confidence": "high", "source": item.get("source", "unknown")})
    valid.sort(key=lambda item: item["start_seconds"])
    merged = []
    for item in valid:
        if merged and item["start_seconds"] <= merged[-1]["end_seconds"] + .35:
            merged[-1]["end_seconds"] = max(merged[-1]["end_seconds"], item["end_seconds"])
            merged[-1]["resume_text"] = resume_text(segments, merged[-1]["end_seconds"])
            merged[-1]["category"] = " / ".join(dict.fromkeys([merged[-1]["category"], item["category"]]))
        else: merged.append(item)
    return merged


def analyze_job(job_id: str) -> None:
    job_dir = JOBS_DIR / job_id
    video = job_dir / "input.mp4"
    audio = job_dir / "audio.wav"
    started_at = time.time()
    try:
        env = environment()["tools"]
        missing = [name for name in ("ffmpeg", "ffprobe", "whisper", "whisper_model") if not env[name]["ready"]]
        if missing: raise RuntimeError("必要なローカルツールが揃っていません。実行環境を確認してください。")
        update(job_id, phase="probing", progress=18)
        metadata = probe_video(video, env["ffprobe"]["path"])
        update(job_id, phase="audio", progress=28, **metadata)
        silences = extract_audio_and_silence(video, audio, env["ffmpeg"]["path"])
        update(job_id, phase="transcribe", progress=45)
        segments = transcribe(audio, job_dir, env["whisper"]["path"], whisper_model())
        (job_dir / "segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
        update(job_id, phase="format", progress=92)
        result = {
            "segments": segments,
            "transcript_segments": len(segments),
            "processing_seconds": round(time.time() - started_at, 1),
            "transcription_model": whisper_model().stem.replace("ggml-", ""),
            "language": "ja",
            "warnings": [],
            **metadata,
        }
        (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        update(job_id, status="complete", phase="complete", progress=100, **result)
    except Exception as error:
        update(job_id, status="failed", phase="failed", error=str(error), progress=0)
    finally:
        # 元動画は変更せず、解析用に作ったローカルコピーだけを破棄する。
        audio.unlink(missing_ok=True)
        video.unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "RoughCutAI/0.1"
    def log_message(self, format, *args): print(f"[{self.log_date_time_string()}] {format % args}")
    def json_response(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health": return self.json_response(environment())
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            with LOCK: job = dict(JOBS.get(job_id, {}))
            return self.json_response(job or {"error": "処理情報が見つかりません。"}, 200 if job else 404)
        files = {"/": "index.html", "/index.html": "index.html", "/styles.css": "styles.css", "/transcript.css": "transcript.css", "/app.js": "app.js"}
        filename = files.get(path)
        if not filename: return self.send_error(404)
        data = (ROOT / filename).read_bytes(); content_type = "text/html; charset=utf-8" if filename.endswith("html") else "text/css; charset=utf-8" if filename.endswith("css") else "application/javascript; charset=utf-8"
        self.send_response(200); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_POST(self):
        if urlparse(self.path).path != "/api/jobs": return self.send_error(404)
        # 一部のMac用ブラウザーはBlob送信時にContent-Lengthを公開しない。
        # 画面側がFile.sizeから付与したローカル専用ヘッダーを代替として使う。
        try: size = int(self.headers.get("Content-Length") or self.headers.get("X-File-Size") or "0")
        except ValueError: size = 0
        if size <= 0 or size > MAX_UPLOAD_BYTES: return self.json_response({"error": "動画のファイル容量を確認できません。"}, 400)
        filename = unquote(self.headers.get("X-Filename", "video.mp4"))
        if not filename.lower().endswith(".mp4"): return self.json_response({"error": "MP4動画を選んでください。"}, 400)
        job_id = uuid.uuid4().hex; job_dir = JOBS_DIR / job_id; job_dir.mkdir(parents=True, exist_ok=False); target = job_dir / "input.mp4"
        remaining = size
        with target.open("wb") as output:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk: break
                output.write(chunk); remaining -= len(chunk)
        if remaining: shutil.rmtree(job_dir, ignore_errors=True); return self.json_response({"error": "動画の読み込みが途中で終了しました。"}, 400)
        with LOCK: JOBS[job_id] = {"job_id": job_id, "filename": filename, "status": "processing", "phase": "upload", "progress": 10, "created_at": time.time()}
        threading.Thread(target=analyze_job, args=(job_id,), daemon=True).start()
        self.json_response({"job_id": job_id}, 202)


def main():
    JOBS_DIR.mkdir(parents=True, exist_ok=True); MODELS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"粗カットAI: http://{HOST}:{PORT}")
    print("終了するには Control+C を押してください。")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__": main()
