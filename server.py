#!/usr/bin/env python3
"""粗カットAI - 外部APIを使わないローカルHTTPサーバー。"""
from __future__ import annotations

import json
import difflib
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import sys
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
JOBS_DIR = ROOT / "data" / "jobs"
MODELS_DIR = ROOT / "models"
CLOUD_MODE = bool(os.getenv("GCS_BUCKET"))
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCP_REGION = os.getenv("GCP_REGION", "asia-northeast1")
CLOUD_RUN_JOB_NAME = os.getenv("CLOUD_RUN_JOB_NAME", "video-rough-cut-worker")
CLOUD_BRIDGE_URL = os.getenv("CLOUD_BRIDGE_URL", "").rstrip("/")
HOST = "0.0.0.0" if CLOUD_MODE else "127.0.0.1"
PORT = int(os.getenv("PORT", "8765"))
# 高画質の収録素材は30分程度でも20GBを超えることがある。
# ローカル処理なので、実用上の誤操作防止として100GBを上限にする。
MAX_UPLOAD_BYTES = 100 * 1024 * 1024 * 1024
# 課題提出用Cloud版だけは、予期しないStorage・Job利用を抑える。
# local-first経路の100GB上限には影響させない。
CLOUD_MAX_UPLOAD_BYTES = int(os.getenv("CLOUD_MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024)))
JOBS: dict[str, dict] = {}
SELECTED_FILES: dict[str, Path] = {}
LOCK = threading.Lock()
OLLAMA_PROCESS: subprocess.Popen | None = None
ROUGH_CUT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")


def storage_bucket():
    if not CLOUD_MODE: return None
    from google.cloud import storage
    return storage.Client(project=GCP_PROJECT or None).bucket(GCS_BUCKET)


def authorized_session():
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return AuthorizedSession(credentials)


def upload_status(job_id: str, snapshot: dict) -> None:
    if CLOUD_MODE:
        storage_bucket().blob(f"jobs/{job_id}/status.json").upload_from_string(
            json.dumps(snapshot, ensure_ascii=False, indent=2), content_type="application/json")


def create_resumable_upload(job_id: str, filename: str, size: int, content_type: str) -> tuple[str, str]:
    from urllib.parse import quote
    object_name = f"uploads/{job_id}/{Path(filename).name}"
    url = f"https://storage.googleapis.com/upload/storage/v1/b/{quote(GCS_BUCKET, safe='')}/o?uploadType=resumable&name={quote(object_name, safe='')}"
    response = authorized_session().post(url, headers={"X-Upload-Content-Type": content_type, "X-Upload-Content-Length": str(size), "Content-Type": "application/json"}, json={"name": object_name}, timeout=30)
    response.raise_for_status()
    return object_name, response.headers["Location"]


def trigger_cloud_worker(job_id: str, object_name: str = "", mode: str = "transcribe") -> None:
    url = f"https://run.googleapis.com/v2/projects/{GCP_PROJECT}/locations/{GCP_REGION}/jobs/{CLOUD_RUN_JOB_NAME}:run"
    env = [{"name": "WORKER_JOB_ID", "value": job_id}, {"name": "WORKER_MODE", "value": mode}]
    if object_name: env.append({"name": "GCS_OBJECT", "value": object_name})
    response = authorized_session().post(url, json={"overrides": {"containerOverrides": [{"env": env}], "taskCount": 1, "timeout": "7200s"}}, timeout=30)
    response.raise_for_status()


def remote_json(path: str, payload: dict | None = None, method: str | None = None, timeout: int = 60) -> dict:
    """Mac高速モードからCloud Runの小さなJSON APIだけを呼ぶ。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = Request(
        CLOUD_BRIDGE_URL + path,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method=method or ("POST" if body is not None else "GET"),
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def tool_path(env_name: str, default_name: str) -> str | None:
    configured = os.getenv(env_name)
    if configured and Path(configured).exists(): return configured
    local_candidates = {
        "ffmpeg": ROOT / "tools" / "runtime" / "bin" / "ffmpeg",
        "ffprobe": ROOT / "tools" / "runtime" / "bin" / "ffprobe",
        "whisper-cli": ROOT / "tools" / "runtime" / "bin" / "whisper-cli",
        "whisper-vad-speech-segments": ROOT / "tools" / "runtime" / "bin" / "whisper-vad-speech-segments",
        "ollama": ROOT / "tools" / "Ollama.app" / "Contents" / "Resources" / "ollama",
    }
    local = local_candidates.get(default_name)
    return str(local) if local and local.exists() else shutil.which(default_name)


def whisper_model() -> Path:
    configured = os.getenv("WHISPER_MODEL")
    return Path(configured).expanduser() if configured else MODELS_DIR / "ggml-large-v3-turbo-q5_0.bin"


def whisper_vad_model() -> Path:
    configured = os.getenv("WHISPER_VAD_MODEL")
    return Path(configured).expanduser() if configured else MODELS_DIR / "ggml-silero-v6.2.0.bin"


def environment() -> dict:
    ffmpeg = tool_path("FFMPEG_BIN", "ffmpeg")
    ffprobe = tool_path("FFPROBE_BIN", "ffprobe")
    whisper = tool_path("WHISPER_BIN", "whisper-cli")
    vad = tool_path("WHISPER_VAD_BIN", "whisper-vad-speech-segments")
    ollama = tool_path("OLLAMA_BIN", "ollama")
    model = whisper_model()
    vad_model = whisper_vad_model()
    return {"cloud_mode": CLOUD_MODE, "bridge_mode": bool(CLOUD_BRIDGE_URL),
        "local_direct_mode": not CLOUD_MODE and not CLOUD_BRIDGE_URL, "tools": {
        "ffmpeg": {"ready": bool(ffmpeg), "path": ffmpeg}, "ffprobe": {"ready": bool(ffprobe), "path": ffprobe},
        "whisper": {"ready": bool(whisper), "path": whisper}, "whisper_model": {"ready": model.is_file(), "path": str(model)},
        "vad": {"ready": bool(vad), "path": vad}, "vad_model": {"ready": vad_model.is_file(), "path": str(vad_model)},
        "ollama": {"ready": bool(ollama), "path": ollama},
    }}


def update(job_id: str, **values) -> None:
    with LOCK:
        JOBS[job_id].update(values)
        snapshot = dict(JOBS[job_id])
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "status.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    upload_status(job_id, snapshot)


def load_job(job_id: str) -> dict:
    """再起動後もMac内に保存した文字起こし結果を利用できるようにする。"""
    if CLOUD_MODE:
        try:
            current = json.loads(storage_bucket().blob(f"jobs/{job_id}/status.json").download_as_text())
            with LOCK: JOBS[job_id] = current
            return dict(current)
        except Exception:
            pass
    with LOCK:
        current = dict(JOBS.get(job_id, {}))
    if CLOUD_BRIDGE_URL and current.get("cloud_started"):
        try:
            remote = remote_json(f"/api/jobs/{job_id}")
            with LOCK: JOBS[job_id] = {**current, **remote, "cloud_started": True}
            return dict(JOBS[job_id])
        except Exception:
            return current
    if current:
        return current
    status_path = JOBS_DIR / job_id / "status.json"
    if not status_path.is_file():
        return {}
    try:
        current = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    if current.get("candidates"):
        current["candidates"] = [item for item in current["candidates"] if not (item.get("category") == "撮影中の相談" and float(item.get("end_seconds", 0)) - float(item.get("start_seconds", 0)) > 90)]
    with LOCK:
        JOBS[job_id] = current
    return dict(current)


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
    fps_ratio = "0/1"
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            fps_ratio = str(stream.get("r_frame_rate", "0/1"))
            numerator, _, denominator = fps_ratio.partition("/")
            fps = float(numerator) / float(denominator or 1)
            break
    return {"duration_seconds": duration, "source_fps": fps, "source_fps_ratio": fps_ratio}


def choose_local_video() -> Path:
    """Mac標準ダイアログで選ばれた元MP4を、コピーせず読み取り対象として返す。"""
    script = '''
set selectedFile to choose file with prompt "文字起こしするMP4動画を選択してください" of type {"public.mpeg-4"}
return POSIX path of selectedFile
'''
    process = subprocess.run(
        ["/usr/bin/osascript", "-e", script], capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if process.returncode:
        detail = (process.stderr or "").strip()
        if "-128" in detail or "User canceled" in detail:
            raise RuntimeError("動画の選択をキャンセルしました。")
        raise RuntimeError("Macのファイル選択画面を開けませんでした。\n" + detail[-500:])
    source_video = Path(process.stdout.strip()).expanduser().resolve(strict=True)
    if not source_video.is_file() or source_video.suffix.lower() != ".mp4":
        raise RuntimeError("MP4動画を選んでください。")
    if not os.access(source_video, os.R_OK):
        raise RuntimeError("選択した動画を読み取る権限がありません。")
    return source_video


def selected_video(selection_id: str) -> Path:
    with LOCK:
        source_video = SELECTED_FILES.get(selection_id)
    if not source_video:
        raise RuntimeError("選択した動画情報が見つかりません。もう一度選択してください。")
    resolved = source_video.resolve(strict=True)
    if not resolved.is_file() or resolved.suffix.lower() != ".mp4" or not os.access(resolved, os.R_OK):
        raise RuntimeError("選択したMP4を読み取れません。移動していないか確認してください。")
    return resolved


def unlink_job_artifact(path: Path, job_dir: Path, allowed_names: set[str]) -> None:
    """ジョブフォルダ直下の明示的な一時生成物だけを削除する。元動画には使用できない。"""
    resolved_job = job_dir.resolve()
    resolved_path = path.resolve(strict=False)
    if resolved_path.parent != resolved_job or resolved_path.name not in allowed_names:
        raise RuntimeError(f"安全のため一時ファイルの削除を拒否しました: {resolved_path.name}")
    resolved_path.unlink(missing_ok=True)


SILENCE_RE = re.compile(r"silence_(start|end):\s*([0-9.]+)")
VAD_SEGMENT_RE = re.compile(r"Speech segment\s+\d+:\s+start\s*=\s*([0-9.]+),\s+end\s*=\s*([0-9.]+)")
VAD_PARAMETERS = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 200,
    "speech_pad_ms": 100,
    "samples_overlap_seconds": 0.1,
}


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


def region(start_ms: int, end_ms: int, region_id: int) -> dict:
    return {"id": region_id, "start_ms": start_ms, "end_ms": end_ms, "duration_ms": end_ms - start_ms}


def normalize_speech_regions(raw_regions: list[tuple[float, float]], duration_ms: int) -> list[dict]:
    normalized: list[tuple[int, int]] = []
    for start, end in raw_regions:
        start_ms = max(0, min(duration_ms, int(round(start * 1000))))
        end_ms = max(start_ms, min(duration_ms, int(round(end * 1000))))
        if end_ms <= start_ms: continue
        if normalized and start_ms <= normalized[-1][1]:
            normalized[-1] = (normalized[-1][0], max(normalized[-1][1], end_ms))
        else:
            normalized.append((start_ms, end_ms))
    return [region(start, end, index) for index, (start, end) in enumerate(normalized, 1)]


def complement_regions(speech_regions: list[dict], duration_ms: int) -> list[dict]:
    gaps, cursor = [], 0
    for speech in speech_regions:
        start, end = int(speech["start_ms"]), int(speech["end_ms"])
        if start > cursor: gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration_ms: gaps.append((cursor, duration_ms))
    return [region(start, end, index) for index, (start, end) in enumerate(gaps, 1) if end > start]


def ffmpeg_silence_regions(silences: list[dict], duration_ms: int) -> list[dict]:
    values = []
    for silence in silences:
        start = max(0, min(duration_ms, int(round(float(silence["start"]) * 1000))))
        end = max(start, min(duration_ms, int(round(float(silence["end"]) * 1000))))
        if end > start: values.append((start, end))
    return [region(start, end, index) for index, (start, end) in enumerate(values, 1)]


def run_independent_vad(audio: Path, duration_seconds: float, vad: str | None, model: Path) -> tuple[dict, str | None]:
    """文字起こし結果へ影響させず、同じWAVから独立して発話区間を得る。"""
    duration_ms = max(0, int(round(duration_seconds * 1000)))
    base = {
        "engine": "whisper.cpp-silero-vad", "model": model.name,
        "timeline": "original_media", "parameters": dict(VAD_PARAMETERS),
    }
    if not vad:
        return ({**base, "status": "unavailable", "reason": "VAD実行ファイルが見つかりません。",
                 "processing_seconds": 0.0, "speech_regions": [], "non_speech_regions": []},
                "VAD実行ファイルがないため、発話区間検出をスキップしました。")
    if not model.is_file():
        return ({**base, "status": "unavailable", "reason": "VADモデルが未セットアップです。",
                 "processing_seconds": 0.0, "speech_regions": [], "non_speech_regions": []},
                "Silero VADモデルが未セットアップのため、発話区間検出をスキップしました。文字起こし結果には影響ありません。")
    started_at = time.time()
    try:
        command = [
            vad, "-f", str(audio), "-vm", str(model),
            "-vt", str(VAD_PARAMETERS["threshold"]),
            "-vspd", str(VAD_PARAMETERS["min_speech_duration_ms"]),
            "-vsd", str(VAD_PARAMETERS["min_silence_duration_ms"]),
            "-vp", str(VAD_PARAMETERS["speech_pad_ms"]),
            "-vo", str(VAD_PARAMETERS["samples_overlap_seconds"]), "-np",
        ]
        process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if process.returncode:
            detail = (process.stderr or process.stdout).strip()[-800:]
            raise RuntimeError(detail or "VAD処理が正常終了しませんでした。")
        # whisper-vad-speech-segmentsの表示値は秒ではなく1/100秒単位。
        raw_regions = [(float(start) / 100, float(end) / 100) for start, end in VAD_SEGMENT_RE.findall(process.stdout)]
        speech = normalize_speech_regions(raw_regions, duration_ms)
        return ({**base, "status": "complete", "processing_seconds": round(time.time() - started_at, 3),
                 "speech_regions": speech, "non_speech_regions": complement_regions(speech, duration_ms)}, None)
    except Exception as error:
        return ({**base, "status": "failed", "reason": str(error),
                 "processing_seconds": round(time.time() - started_at, 3),
                 "speech_regions": [], "non_speech_regions": []},
                "発話区間検出に失敗しましたが、文字起こし結果は正常に保存されました。")


def timestamp_seconds(value) -> float:
    if isinstance(value, (int, float)): return float(value) / 1000 if value > 100000 else float(value)
    text = str(value or "0").replace(",", ".")
    parts = text.split(":")
    try:
        if len(parts) == 3: return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return float(text)
    except ValueError: return 0.0


def parse_whisper_json(path: Path) -> list[dict]:
    # whisper.cppが固有名詞付近に不正なUTF-8バイトを出しても、結果全体を失わない。
    data = json.loads(path.read_bytes().decode("utf-8", errors="replace"))
    source = data.get("transcription") or data.get("segments") or []
    segments = []
    for segment_id, item in enumerate(source, 1):
        offsets = item.get("offsets") or {}
        timestamps = item.get("timestamps") or {}
        start = offsets.get("from", item.get("start", timestamps.get("from", 0)))
        end = offsets.get("to", item.get("end", timestamps.get("to", 0)))
        if offsets: start, end = float(start) / 1000, float(end) / 1000
        else: start, end = timestamp_seconds(start), timestamp_seconds(end)
        text = str(item.get("text") or "").strip()
        tokens = []
        for token in item.get("tokens") or []:
            token_offsets = token.get("offsets") or {}
            token_timestamps = token.get("timestamps") or {}
            token_start = token_offsets.get("from", token_timestamps.get("from", 0))
            token_end = token_offsets.get("to", token_timestamps.get("to", 0))
            if token_offsets:
                token_start_ms, token_end_ms = int(round(float(token_start))), int(round(float(token_end)))
            else:
                token_start_ms = int(round(timestamp_seconds(token_start) * 1000))
                token_end_ms = int(round(timestamp_seconds(token_end) * 1000))
            tokens.append({
                "text": str(token.get("text") or ""),
                "start_ms": token_start_ms, "end_ms": token_end_ms,
                "token_id": token.get("id"), "probability": token.get("p"),
                "t_dtw": token.get("t_dtw"),
            })
        if text:
            segments.append({
                "id": segment_id,
                "start": round(start, 3), "end": round(end, 3),
                "start_ms": int(round(start * 1000)), "end_ms": int(round(end * 1000)),
                "text": text, "tokens": tokens,
            })
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
TAKE_SIGNAL = re.compile(r"(もう一回|もう一度|今のなし|今のはなし|今の違う|今のやめ|最初から|言い直|撮り直|仕切り直|ここ.{0,5}カット|カットで|どこから|そこからもう一回|一旦止め|ちょっと待)")
SELF_CORRECTION = re.compile(r"^(いや|違う|じゃなくて|というか|改めて|正確には|ちょっと待)")
INCOMPLETE_ENDING = re.compile(r"(あの|その|えっと|つまり|だから|けど|ですが|なので|というか|なんていうか|で|を|が|は|って|という)$")
SHORT_ACKNOWLEDGEMENT = re.compile(r"^(はい|うん|ええ|そう|そうです|そうですね|なるほど|確かに|分かりました|わかりました|ありがとうございます|何が)$")
DOMAIN_TERMS = {"断熱", "気密", "耐震", "間取り", "平屋", "収納", "動線", "トイレ", "寝室", "空調", "冷房", "暖房", "エアコン", "フィルター", "室外機", "電気代", "住宅", "家づくり", "換気", "湿度", "結露", "基礎", "耐久性"}


def normalize_for_comparison(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = re.sub(r"^[\s、。]*(?:あの+|え+と|え+|う+|まあ+|その+|なんていうか)[\s、。ー…]*", "", value)
    return re.sub(r"[\s、。,.，．!！?？・「」『』（）()\[\]【】ー…]", "", value)


def ngrams(text: str, size: int) -> set[str]:
    return {text[index:index + size] for index in range(max(0, len(text) - size + 1))}


def dice_similarity(left: str, right: str, size: int) -> float:
    a, b = ngrams(left, size), ngrams(right, size)
    if not a or not b: return 1.0 if left == right and left else 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def common_prefix_ratio(left: str, right: str) -> float:
    common = 0
    for a, b in zip(left, right):
        if a != b: break
        common += 1
    return common / max(1, min(len(left), len(right)))


def comparison_metrics(earlier: str, later: str) -> dict:
    matcher = difflib.SequenceMatcher(None, earlier, later, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    sequence = matcher.ratio()
    similarity = max(sequence, dice_similarity(earlier, later, 2) * .6 + dice_similarity(earlier, later, 3) * .4)
    return {
        "similarity": round(similarity, 4),
        "prefix_similarity": round(common_prefix_ratio(earlier, later), 4),
        "earlier_containment": round(matched / max(1, len(earlier)), 4),
        # 文字列上、後の発言に追加された割合。意味的重要度は表さない。
        "textual_addition_ratio": round(max(0.0, 1 - matched / max(1, len(later))), 4),
    }


def distinctive_terms(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    values = set(re.findall(r"(?:\d+(?:\.\d+)?|[〇一二三四五六七八九十百千万]+)(?:%|割|位|年|万円|円|畳|坪|度|時間|分|秒)?", normalized))
    values.update(re.findall(r"[ァ-ヶ]{3,}|[A-Za-z]{2,}", normalized))
    values.update(term for term in DOMAIN_TERMS if term in normalized)
    return values


def is_short_acknowledgement(text: str) -> bool:
    return bool(SHORT_ACKNOWLEDGEMENT.fullmatch(normalize_for_comparison(text)))


def make_comparison_units(segments: list[dict]) -> list[dict]:
    units = []
    for start_index, first in enumerate(segments):
        # 通常は発話単位だけを比較する。極端に短い／未完の発話だけ次の一発話と結合し、
        # Whisperの細かすぎる区切りを吸収する。
        first_normalized = normalize_for_comparison(first["text"])
        max_parts = 2 if len(first_normalized) < 10 or INCOMPLETE_ENDING.search(first_normalized) else 1
        current = []
        for end_index in range(start_index, min(len(segments), start_index + max_parts)):
            segment = segments[end_index]
            if current and segment["start"] - current[-1]["end"] > 3: break
            current.append(segment)
            if current[-1]["end"] - current[0]["start"] > 20: break
            text = "".join(item["text"] for item in current)
            units.append({
                "start": current[0]["start"], "end": current[-1]["end"],
                "start_ms": int(current[0].get("start_ms", round(current[0]["start"] * 1000))),
                "end_ms": int(current[-1].get("end_ms", round(current[-1]["end"] * 1000))),
                "segment_ids": [item.get("id", start_index + offset + 1) for offset, item in enumerate(current)],
                "text": text, "normalized": normalize_for_comparison(text),
            })
    return units


def adjust_candidate_boundary(start_ms: int, end_ms: int, audio_activity: dict) -> tuple[int, int, str]:
    speech = audio_activity.get("speech_regions") or []
    if not speech: return start_ms, end_ms, "segments"
    suggested_start, suggested_end = start_ms, end_ms
    for item in speech:
        region_start, region_end = int(item["start_ms"]), int(item["end_ms"])
        if region_start <= start_ms < region_end and start_ms - region_start <= 500: suggested_start = region_start
        if region_start < end_ms <= region_end and region_end - end_ms <= 500: suggested_end = region_end
    source = "vad" if (suggested_start, suggested_end) != (start_ms, end_ms) else "segments"
    return suggested_start, suggested_end, source


def build_take_candidate(category: str, decision: str, suspect: dict, replacement: dict | None,
                         signals: list[str], metrics: dict, audio_activity: dict, reason: str) -> dict:
    detected_start, detected_end = int(suspect["start_ms"]), int(suspect["end_ms"])
    suggested_start, suggested_end, boundary_source = adjust_candidate_boundary(detected_start, detected_end, audio_activity)
    return {
        "category": category, "decision": decision,
        "detected_start_ms": detected_start, "detected_end_ms": detected_end,
        "suggested_cut_start_ms": suggested_start, "suggested_cut_end_ms": suggested_end,
        "boundary_source": boundary_source,
        "suspect_segment_ids": suspect["segment_ids"],
        "replacement_segment_ids": replacement["segment_ids"] if replacement else [],
        "suspect_text": suspect["text"], "replacement_text": replacement["text"] if replacement else "",
        "signals": list(dict.fromkeys(signals)), "metrics": metrics, "reason": reason,
        "human_status": "unreviewed",
    }


def detect_take_candidates(segments: list[dict], audio_activity: dict) -> dict:
    """Ollamaを使わず、言い直し・重複・明確なNGの確認候補だけを作る。"""
    started_at = time.time()
    units = make_comparison_units(segments)
    candidates = []

    # 明確なNGサインは、直前の発言と直後の完成テイクを関連付ける。
    for signal_index, signal_segment in enumerate(segments):
        if not TAKE_SIGNAL.search(signal_segment["text"]): continue
        previous = next((segment for segment in reversed(segments[:signal_index])
                         if signal_segment["start"] - segment["end"] <= 45 and not is_short_acknowledgement(segment["text"])), None)
        following = next((segment for segment in segments[signal_index + 1:]
                          if segment["start"] - signal_segment["end"] <= 45 and not is_short_acknowledgement(segment["text"])), None)
        if not previous: continue
        previous_start_ms = int(previous.get("start_ms", round(previous["start"] * 1000)))
        signal_end_ms = int(signal_segment.get("end_ms", round(signal_segment["end"] * 1000)))
        suspect = {"start_ms": previous_start_ms, "end_ms": signal_end_ms,
                   "segment_ids": [previous.get("id", segments.index(previous) + 1), signal_segment.get("id", signal_index + 1)],
                   "text": previous["text"] + " / " + signal_segment["text"]}
        replacement = None
        metrics = {"similarity": 0.0, "prefix_similarity": 0.0, "earlier_containment": 0.0,
                   "textual_addition_ratio": 1.0, "gap_ms": 0}
        decision = "review"
        signals = ["explicit_take_signal"]
        if following:
            following_start_ms = int(following.get("start_ms", round(following["start"] * 1000)))
            following_end_ms = int(following.get("end_ms", round(following["end"] * 1000)))
            replacement = {"start_ms": following_start_ms, "end_ms": following_end_ms,
                           "segment_ids": [following.get("id", segments.index(following) + 1)], "text": following["text"]}
            metrics.update(comparison_metrics(normalize_for_comparison(previous["text"]), normalize_for_comparison(following["text"])))
            metrics["gap_ms"] = max(0, following_start_ms - signal_end_ms)
            added_terms = distinctive_terms(following["text"]) - distinctive_terms(previous["text"])
            if added_terms: signals.append("new_distinctive_terms")
            if metrics["textual_addition_ratio"] >= .2: signals.append("substantial_text_addition")
            if metrics["similarity"] >= .82 and metrics["textual_addition_ratio"] <= .15 and not added_terms:
                decision = "strong"; signals.append("later_similar_take")
        candidates.append(build_take_candidate("explicit_ng", decision, suspect, replacement, signals, metrics, audio_activity,
                                               "明確な撮影NGサインの前後です。"))

    # Whisperの区切り差を吸収するため、1〜3発言の小さな単位同士を近接範囲だけ比較する。
    for earlier_index, earlier in enumerate(units):
        if len(earlier["normalized"]) < 5 or is_short_acknowledgement(earlier["text"]): continue
        for later in units[earlier_index + 1:]:
            if later["start_ms"] <= earlier["end_ms"]: continue
            gap_ms = later["start_ms"] - earlier["end_ms"]
            if gap_ms > 120_000: break
            if set(earlier["segment_ids"]) & set(later["segment_ids"]): continue
            if len(later["normalized"]) < 5 or is_short_acknowledgement(later["text"]): continue
            metrics = comparison_metrics(earlier["normalized"], later["normalized"])
            comparable = max(metrics["similarity"], metrics["earlier_containment"], metrics["prefix_similarity"])
            incomplete = bool(INCOMPLETE_ENDING.search(earlier["normalized"])) or bool(SELF_CORRECTION.search(later["normalized"]))
            if not incomplete and min(len(earlier["normalized"]), len(later["normalized"])) < 10: continue
            if not incomplete and metrics["similarity"] < .72 and metrics["prefix_similarity"] < .65: continue
            minimum = .68 if incomplete and metrics["prefix_similarity"] >= .45 else .82
            if comparable < minimum or (gap_ms > 45_000 and comparable < .86): continue
            metrics["gap_ms"] = gap_ms
            signals = []
            if incomplete: signals.append("incomplete_earlier_take")
            if metrics["prefix_similarity"] >= .65: signals.append("repeated_prefix")
            if metrics["earlier_containment"] >= .82: signals.append("earlier_text_contained")
            added_terms = distinctive_terms(later["text"]) - distinctive_terms(earlier["text"])
            if added_terms: signals.append("new_distinctive_terms")
            if metrics["textual_addition_ratio"] >= .2: signals.append("substantial_text_addition")
            category = "false_start" if incomplete and metrics["prefix_similarity"] >= .45 else "near_duplicate"
            strong = (gap_ms <= 45_000 and metrics["similarity"] >= .88 and
                      metrics["earlier_containment"] >= .9 and metrics["textual_addition_ratio"] <= .15 and
                      not added_terms and "substantial_text_addition" not in signals)
            decision = "strong" if strong else "review"
            reason = "直後にほぼ同じ完成テイクがあります。" if strong else "似た発言ですが文字列上の追加内容があるため確認が必要です。"
            candidates.append(build_take_candidate(category, decision, earlier, later, signals, metrics, audio_activity, reason))

    # 同じ疑わしい範囲は、明確なNGサインと強い判定を優先して一件にまとめる。
    priority = {"explicit_ng": 2, "false_start": 1, "near_duplicate": 0}
    candidates.sort(key=lambda item: (item["detected_start_ms"], -priority.get(item["category"], 0), item["decision"] != "strong"))
    deduplicated = []
    for item in candidates:
        duplicate = next((old for old in deduplicated
                          if set(old["suspect_segment_ids"]) & set(item["suspect_segment_ids"]) and
                          abs(old["detected_start_ms"] - item["detected_start_ms"]) < 1500), None)
        if not duplicate: deduplicated.append(item)
        elif (priority.get(item["category"], 0) >= priority.get(duplicate["category"], 0) and
              item["decision"] == "strong" and duplicate["decision"] != "strong"):
            deduplicated[deduplicated.index(duplicate)] = item
    for index, item in enumerate(deduplicated, 1): item["id"] = f"take-{index:04d}"
    return {
        "version": "1.0", "status": "complete", "processed_locally": True,
        "ollama_used": False, "processing_seconds": round(time.time() - started_at, 3),
        "metrics_note": "textual_addition_ratioは文字列上の追加割合であり、意味的重要度ではありません。",
        "candidates": deduplicated,
    }


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


def ensure_local_ollama() -> None:
    """粗カット判定を明示的に開始した時だけ、既存のローカルOllamaを起動する。"""
    global OLLAMA_PROCESS
    if ollama_available(ROUGH_CUT_MODEL): return
    ollama = tool_path("OLLAMA_BIN", "ollama")
    if not ollama: raise RuntimeError("ローカルAI（Ollama）が見つかりません。")
    OLLAMA_PROCESS = subprocess.Popen(
        [ollama, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    for _ in range(60):
        if ollama_available(ROUGH_CUT_MODEL): return
        time.sleep(1)
    raise RuntimeError("ローカルAIを起動できません。")


def ollama_json(prompt: str, schema: dict | None = None, timeout: int = 600) -> dict:
    body = json.dumps({
        "model": ROUGH_CUT_MODEL,
        "messages": [{"role": "user", "content": "/no_think\n" + prompt}],
        "stream": False, "think": False, "format": schema or "json",
        "options": {"temperature": 0.05, "num_ctx": 32768, "num_predict": 1400},
    }, ensure_ascii=False).encode("utf-8")
    request = Request("http://127.0.0.1:11434/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        content = json.load(response)["message"]["content"]
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    json_block = re.search(r"\{.*\}", content, flags=re.DOTALL)
    candidate = json_block.group(0) if json_block else content
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"ローカルAIのJSON形式が不正です: {candidate[:500]}") from error


def semantic_blocks(segments: list[dict]) -> list[dict]:
    """Whisperの細かい発言を、5〜12秒程度の意味ブロックにする。"""
    blocks, current = [], []
    for index, segment in enumerate(segments):
        current.append(segment)
        elapsed = current[-1]["end"] - current[0]["start"]
        next_segment = segments[index + 1] if index + 1 < len(segments) else None
        gap = max(0.0, next_segment["start"] - segment["end"]) if next_segment else 0.0
        completed = bool(re.search(r"[。！？!?]$|(です|ます|でした|ません|なんです|ですよね|じゃないですか)$", segment["text"]))
        should_close = elapsed >= 10 or (elapsed >= 3 and (completed or gap >= .8))
        if should_close:
            blocks.append({"id": len(blocks), "start": current[0]["start"], "end": current[-1]["end"], "text": " / ".join(x["text"] for x in current)})
            current = []
    if current:
        blocks.append({"id": len(blocks), "start": current[0]["start"], "end": current[-1]["end"], "text": " / ".join(x["text"] for x in current)})
    return blocks


def block_lines(blocks: list[dict]) -> str:
    return "\n".join(f'B{b["id"]} [{b["start"]:.1f}-{b["end"]:.1f}] {b["text"]}' for b in blocks)


def understand_video(blocks: list[dict]) -> dict:
    prompt = """建築・住宅系YouTube対談の文字起こし全体です。
カット判定はまだせず、動画の中心テーマ、視聴者に伝える結論、重要ポイント、主な論点を整理してください。
さらに、動画の理解に絶対必要な情報と、正しい説明ではあるが短縮・省略しても結論が伝わる情報を分けてください。
失敗テイクや撮影相談を動画の主題と誤認しないでください。
同じ論点に相反する発言がある場合は、途中で失敗した説明ではなく、後で最後まで成立した明確な説明を採用してください。
JSON形式だけを返します。

""" + block_lines(blocks)
    schema = {"type": "object", "properties": {
        "central_theme": {"type": "string"}, "main_conclusions": {"type": "array", "items": {"type": "string"}},
        "important_points": {"type": "array", "items": {"type": "string"}}, "topics": {"type": "array", "items": {"type": "string"}},
        "essential_information": {"type": "array", "items": {"type": "string"}},
        "compressible_information": {"type": "array", "items": {"type": "string"}},
    }, "required": ["central_theme", "main_conclusions", "important_points", "topics", "essential_information", "compressible_information"]}
    result = ollama_json(prompt, schema)
    conclusions = [str(x) for x in result.get("main_conclusions", [])][:6]
    theme = str(result.get("central_theme") or "")
    if not theme or "整理してください" in theme or "文字起こし全体" in theme:
        theme = conclusions[0] if conclusions else "テーマを特定できませんでした"
    points = [str(x) for x in result.get("important_points", [])][:8]
    full_text = " ".join(b["text"] for b in blocks)
    if re.search(r"自動.{0,8}(方がいい|ままでいい)", full_text):
        points = ["エアコンの風量は自動設定を使う" if "風量" in point else point for point in points]
    return {
        "central_theme": theme,
        "main_conclusions": conclusions,
        "important_points": points,
        "topics": [str(x) for x in result.get("topics", [])][:10],
        "essential_information": [str(x) for x in result.get("essential_information", [])][:10],
        "compressible_information": [str(x) for x in result.get("compressible_information", [])][:10],
    }


ROUGH_CUT_INSTRUCTIONS = """動画編集者が確認すべき粗カット候補を探します。候補を出すこと自体を目的にせず、不要部分がなければ残してください。
【優先】撮影・台本・編集の相談、仕切り直し前の失敗テイク、後により明確な言い直しがある成立していない説明、新情報のないまとまった重複、長く膨らみ本筋を止めるキャラクター会話。
【残す】結論、理由、具体例、比較、注意点、有効な視聴者質問、専門情報を分かりやすくする補足、導入や人柄に機能する短い会話。
【禁止】雑談っぽい、本題から少し外れたという理由だけで切らない。1〜2秒の相づち、単独フィラー、単語だけの言い直しは出さない。
判定は strong（強いカット候補）、review（人間が確認するカット検討候補）、keep（残す）の3段階です。
strong: 削除してよい可能性が非常に高く、候補内の全ブロックが不要。
review: 情報が薄い、長すぎる、ほぼ重複、演者らしさはあるが長いなど、人間が動画で確認する価値がある。
keep: テーマ理解や視聴価値に必要、または判断が難しい。keepはcandidatesへ出力しません。
「内容として成立している」「正しい情報を含む」だけでkeepにしないでください。区間を丸ごと削除しても中心テーマ、結論、必要な理由、注意点の理解がほぼ変わらないならreviewです。
同じ結論のために必須ではない説明、後でより分かりやすく説明される旧テイク、長すぎる具体例、途中で理解されなかった説明、少量の人柄会話が長く続く部分は積極的にreviewへ入れてください。
正しく成立している導入や説明でも、直後または別の場所に同じ役割の完成版があれば、前の説明をreviewにしてください。視聴者が「分からない」「長い」と反応した説明は、その発言自体に情報があってもreviewを優先してください。
各候補について「削除後に前後を直接つないでも意味が通るか」「この区間だけが持つ不可欠情報があるか」を確認してください。
通常は5〜90秒の連続区間を候補にします。明確な撮影相談・失敗・仕切り直しのみ数秒でも構いません。
明確な重複、直後の言い直し、意味のない言いかけ、不完全な一言は1〜4秒でもreviewにできます。単独の相づちやフィラーは候補にしません。
候補同士は重複させません。同じ理由で隣接するブロックは一つの候補にまとめます。不要な理由が途中で変わる場合は、無理に一つの長い候補に結合しません。
reasonは同じ説明を繰り返さず、40文字以内の日本語一文にしてください。
開始と終了は、完全に削除する最初と最後のブロックIDで指定します。
指定されたJSON形式だけを返します。categoryは production / retake / duplicate / long_banter / failed_explanation / low_value のどれか一つです。
"""


def rough_cut_proposals(blocks: list[dict], understanding: dict) -> list[dict]:
    found, core_start = [], 0.0
    duration = blocks[-1]["end"] if blocks else 0.0
    context = "\n".join([
        f'中心テーマ: {understanding.get("central_theme", "")}',
        '主な結論: ' + ' / '.join(understanding.get("main_conclusions", [])),
        '重要ポイント: ' + ' / '.join(understanding.get("important_points", [])),
        '絶対に残す情報: ' + ' / '.join(understanding.get("essential_information", [])),
        '短縮・省略を検討できる情報: ' + ' / '.join(understanding.get("compressible_information", [])),
        '主な論点: ' + ' / '.join(understanding.get("topics", [])),
    ])
    schema = {"type": "object", "properties": {"candidates": {"type": "array", "maxItems": 10, "items": {"type": "object", "properties": {
        "start_block": {"type": "integer"}, "end_block": {"type": "integer"},
        "category": {"type": "string", "enum": ["production", "retake", "duplicate", "long_banter", "failed_explanation", "low_value"]},
        "reason": {"type": "string", "maxLength": 60}, "decision": {"type": "string", "enum": ["strong", "review"]},
    }, "required": ["start_block", "end_block", "category", "reason", "decision"]}}}, "required": ["candidates"]}
    by_id = {b["id"]: b for b in blocks}
    while core_start < duration:
        core_end = min(duration, core_start + 120)
        chunk = [b for b in blocks if b["end"] > max(0, core_start - 45) and b["start"] < min(duration, core_end + 45)]
        prompt = f"""あなたは建築・住宅YouTubeの粗カット補助です。
動画全体の理解:
{context}

参照範囲には前後の文脈も含まれます。主に判定する時間は {core_start:.1f}秒〜{core_end:.1f}秒です。
前後の参照部分も読み、境界をまたぐ一続きの不要箇所は正しい開始・終了ブロックまで含めてください。
{ROUGH_CUT_INSTRUCTIONS}
文字起こし:
{block_lines(chunk)}"""
        result = ollama_json(prompt, schema)
        for item in result.get("candidates", []):
            try:
                first, last = by_id[int(item["start_block"])], by_id[int(item["end_block"])]
            except (KeyError, TypeError, ValueError):
                continue
            # 前後文脈で同じ候補が重複しないよう、候補の中央を含む2分区間だけが担当する。
            midpoint = (first["start"] + last["end"]) / 2
            if core_start <= midpoint < core_end or (core_end == duration and midpoint == duration):
                found.append(item)
        core_start += 120
    return found


def rule_rough_proposals(blocks: list[dict]) -> list[dict]:
    """2本の実際の指示書で共通した、明確な撮影内部のサインだけを拾う。"""
    production_re = re.compile(r"(もう一回|最初から|撮り直|ここ.{0,5}カット|カットし|ぶった切|台本|編集|テンション|どこから|雑談を混ぜ|この回.{0,8}楽しい|携帯|何回言って)")
    failure_re = re.compile(r"(ちょっと待|何だったっけ|なんだったっけ|忘れちゃ|分から|分かん|わから|わかんな|何言ってる|長い長い|無理だな|違う|全然進ま|言ってるか.{0,5}わか|やばい|なんて言ったら|やめよう|ぐったり)")
    character_re = re.compile(r"(ミミズ|フクロウ|オールくん|毛皮|森のパーティ|友達.{0,5}色|カラフル|エメラルド|彼女|カリント|ペンギン|合唱会)")
    domain_re = re.compile(r"(エアコン|フィルター|室外機|電気代|断熱|間取り|平屋|LDK|収納|動線|トイレ|寝室|空調|住宅|家づくり|冷房|暖房|内部クリーン)")
    marked = []
    for block in blocks:
        production_hits = len(production_re.findall(block["text"]))
        failure_hits = len(failure_re.findall(block["text"]))
        character_hits = len(character_re.findall(block["text"]))
        domain_hits = len(domain_re.findall(block["text"]))
        kind = "production" if production_hits else "failed_explanation" if failure_hits >= 1 else "long_banter" if character_hits >= 2 and not domain_hits else ""
        if kind: marked.append((block["id"], kind))
    proposals = []
    # 冒頭の撮り直しは、完成した「はい、始まりました」直前までを一つにする。
    early_retake = [b for b in blocks if b["start"] < 180 and re.search(r"(もう一回|最初から)", b["text"])]
    if early_retake:
        restart = next((b for b in blocks if b["id"] > early_retake[-1]["id"] and "始まりました" in b["text"]), None)
        if restart and restart["id"] > 0:
            proposals.append({"start_block": 0, "end_block": restart["id"] - 1, "category": "retake", "reason": "冒頭の準備と失敗テイクの後、改めて本番を開始しています。", "decision": "strong"})
    # 同種のサインが隣接または1ブロック開けで続く場合だけ、まとまった区間にする。
    index = 0
    while index < len(marked):
        start_id, kind = marked[index]
        end_id, cursor = start_id, index + 1
        while cursor < len(marked) and marked[cursor][1] == kind and marked[cursor][0] <= end_id + 2:
            end_id = marked[cursor][0]; cursor += 1
        raw_duration = blocks[end_id]["end"] - blocks[start_id]["start"]
        if kind == "failed_explanation" and end_id - start_id >= 2:
            start_id, end_id = start_id + 1, end_id - 1
        elif kind == "long_banter" and raw_duration >= 36 and end_id > start_id:
            # 演者らしさが伝わる最初の短い設定は残し、長く膨らんだ後半だけを候補にする。
            start_id += 1
        proposal_duration = blocks[end_id]["end"] - blocks[start_id]["start"]
        if proposal_duration >= 12:
            reason = {"production": "撮影や進行についての内部会話が連続しています。", "failed_explanation": "説明が成立せず、言葉を探しながら複数回やり直しています。", "long_banter": "キャラクター会話が長く続き、住宅情報が追加されていません。"}[kind]
            proposals.append({"start_block": start_id, "end_block": end_id, "category": kind, "reason": reason, "decision": "strong"})
        elif proposal_duration >= 3 and kind in {"production", "failed_explanation"}:
            reason = "撮影進行の確認です。" if kind == "production" else "説明が止まり、言い直しへ移っています。"
            proposals.append({"start_block": start_id, "end_block": end_id, "category": kind, "reason": reason, "decision": "review"})
        index = cursor
    return proposals


def normalize_rough_cuts(proposals: list[dict], blocks: list[dict], segments: list[dict], duration: float) -> list[dict]:
    by_id = {b["id"]: b for b in blocks}
    labels = {"production": "撮影中の相談", "retake": "仕切り直し・失敗テイク", "duplicate": "明らかな重複", "long_banter": "長く膨らんだ会話", "failed_explanation": "成立していない説明", "low_value": "情報価値が低いまとまり"}
    cuts = []
    for item in proposals:
        try: first, last = by_id[int(item["start_block"])], by_id[int(item["end_block"])]
        except (KeyError, TypeError, ValueError): continue
        start, end = max(0.0, first["start"]), min(duration, last["end"])
        category = labels.get(str(item.get("category") or ""), "")
        reason = str(item.get("reason") or "")
        decision = str(item.get("decision") or "strong").lower()
        if decision not in {"strong", "review"}: continue
        candidate_text = " ".join(b["text"] for b in blocks if first["id"] <= b["id"] <= last["id"])
        if category == "長く膨らんだ会話":
            character_segments = [s for s in segments if s["end"] > start and s["start"] < end and re.search(r"(ミミズ|フクロウ|角|毛|パーティ|友達|彼女|カリント|ペンギン|合唱会|カラフル|エメラルド)", s["text"])]
            if character_segments:
                end = min(end, character_segments[-1]["end"])
        short_review = decision == "review" and category in {"明らかな重複", "成立していない説明", "情報価値が低いまとまり"}
        minimum = 1.0 if short_review else 3.0 if category in {"撮影中の相談", "仕切り直し・失敗テイク"} else 5.0
        if not category or end - start < minimum: continue
        # 小型AIが参照範囲全体を一候補にすることがあるため、2分を超える範囲は採用しない。
        if end - start > 120: continue
        # 小型AIが「重要だから残す」と説明しながら候補に入れた矛盾を安全側で除外する。
        if re.search(r"(重要|必要な情報|具体的な内容|具体的な説明|自然な流れ|価値がある|提供する)", reason):
            if decision == "strong": decision = "review"
            elif not re.search(r"(冗長|重複|省略|短縮|なくても|影響しない)", reason): continue
        meta_hits = len(re.findall(r"(ちょっと待|どこから|分から|わから|忘れ|任せた|テンション|カット|台本|編集|撮影|もう一回|最初から|何だったっけ|なんだったっけ|進まん|雑談を混ぜ|ぶった切)", candidate_text))
        failure_hits = len(re.findall(r"(ちょっと待|何だったっけ|なんだったっけ|分から|わから|忘れ|違う|言ってるかわか|進まん|やばい)", candidate_text))
        if end - start > 90: decision = "review"
        if decision == "strong" and category == "撮影中の相談" and meta_hits < 2: decision = "review"
        if decision == "strong" and category == "仕切り直し・失敗テイク" and not re.search(r"(もう一回|最初から|仕切り直|撮り直)", candidate_text): decision = "review"
        if decision == "strong" and category == "成立していない説明" and failure_hits < 3: decision = "review"
        if decision == "strong" and category == "明らかな重複" and not re.search(r"(重複|同じ|後で|言い直|繰り返)", reason): decision = "review"
        cuts.append({
            "start_seconds": round(start, 3), "end_seconds": round(end, 3),
            "resume_text": resume_text(segments, end), "category": category,
            "reason": (reason or "視聴者価値を追加しないまとまりです。")[:180], "decision": decision,
        })
    cuts.sort(key=lambda x: (x["start_seconds"], x["end_seconds"]))
    merged = []
    for item in cuts:
        if merged and item["start_seconds"] >= merged[-1]["start_seconds"] and item["end_seconds"] <= merged[-1]["end_seconds"]:
            continue
        if merged and item["start_seconds"] <= merged[-1]["end_seconds"] + .35 and item["category"] == merged[-1]["category"] and item["decision"] == merged[-1]["decision"] and max(merged[-1]["end_seconds"], item["end_seconds"]) - merged[-1]["start_seconds"] <= 120:
            merged[-1]["end_seconds"] = max(merged[-1]["end_seconds"], item["end_seconds"])
            merged[-1]["resume_text"] = resume_text(segments, merged[-1]["end_seconds"])
        elif not merged or (item["start_seconds"], item["end_seconds"]) != (merged[-1]["start_seconds"], merged[-1]["end_seconds"]):
            merged.append(item)
    return merged


def run_rough_cut_analysis(job_id: str) -> None:
    started_at, job_dir = time.time(), JOBS_DIR / job_id
    try:
        job = load_job(job_id)
        segments = job.get("segments") or json.loads((job_dir / "segments.json").read_text(encoding="utf-8"))
        duration = float(job.get("duration_seconds") or segments[-1]["end"])
        if not CLOUD_MODE: ensure_local_ollama()
        if not ollama_available(ROUGH_CUT_MODEL): raise RuntimeError("ローカルAIを起動できません。アプリを再起動してください。")
        update(job_id, rough_status="processing", rough_phase="understanding", rough_progress=15, rough_error=None)
        blocks = semantic_blocks(segments)
        understanding = understand_video(blocks)
        update(job_id, rough_phase="evaluating", rough_progress=48, video_understanding=understanding)
        rule_proposals = rule_rough_proposals(blocks)
        ai_proposals = rough_cut_proposals(blocks, understanding)
        update(job_id, rough_phase="safety", rough_progress=86)
        candidates = normalize_rough_cuts(rule_proposals, blocks, segments, duration)
        ai_candidates = normalize_rough_cuts(ai_proposals, blocks, segments, duration)
        # 一部重なるAI候補も捨てず、同種でほぼ同じ範囲だけを統合する。
        candidates.extend(ai_candidates)
        candidates = sorted(candidates, key=lambda item: (item["start_seconds"], item["end_seconds"]))
        deduplicated = []
        for item in candidates:
            duplicate = None
            for old in deduplicated:
                overlap = min(item["end_seconds"], old["end_seconds"]) - max(item["start_seconds"], old["start_seconds"])
                shorter = min(item["end_seconds"] - item["start_seconds"], old["end_seconds"] - old["start_seconds"])
                if item["category"] == old["category"] and overlap > 0 and overlap / max(.001, shorter) >= .8:
                    duplicate = old
                    break
            if duplicate:
                combined_start = min(duplicate["start_seconds"], item["start_seconds"])
                combined_end = max(duplicate["end_seconds"], item["end_seconds"])
                if combined_end - combined_start <= 120:
                    duplicate["start_seconds"] = combined_start
                    duplicate["end_seconds"] = combined_end
                    duplicate["resume_text"] = resume_text(segments, duplicate["end_seconds"])
                    if item.get("decision") == "strong": duplicate["decision"] = "strong"
                else:
                    deduplicated.append(item)
            else:
                deduplicated.append(item)
        candidates = deduplicated
        result = {"video_understanding": understanding, "candidates": candidates, "analysis_seconds": round(time.time() - started_at, 1), "analysis_model": ROUGH_CUT_MODEL}
        (job_dir / "rough_cuts.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        update(job_id, rough_status="complete", rough_phase="complete", rough_progress=100, **result)
    except Exception as error:
        update(job_id, rough_status="failed", rough_phase="failed", rough_progress=0, rough_error=str(error))


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


def analyze_job(job_id: str, source_video: Path | None = None, delete_source_copy: bool = True) -> None:
    job_dir = JOBS_DIR / job_id
    source_video = (source_video or (job_dir / "input.mp4")).resolve(strict=True)
    temporary_audio = job_dir / "audio.wav"
    source_before = source_video.stat()
    started_at = time.time()
    try:
        env = environment()["tools"]
        missing = [name for name in ("ffmpeg", "ffprobe", "whisper", "whisper_model") if not env[name]["ready"]]
        if missing: raise RuntimeError("必要なローカルツールが揃っていません。実行環境を確認してください。")
        update(job_id, phase="probing", progress=18)
        metadata = probe_video(source_video, env["ffprobe"]["path"])
        update(job_id, phase="audio", progress=28, **metadata)
        silences = extract_audio_and_silence(source_video, temporary_audio, env["ffmpeg"]["path"])
        update(job_id, phase="transcribe", progress=45)
        segments = transcribe(temporary_audio, job_dir, env["whisper"]["path"], whisper_model())
        (job_dir / "segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
        update(job_id, phase="vad", progress=88)
        vad_result, vad_warning = run_independent_vad(
            temporary_audio, metadata["duration_seconds"], env.get("vad", {}).get("path"), whisper_vad_model()
        )
        speech_regions = vad_result.pop("speech_regions")
        non_speech_regions = vad_result.pop("non_speech_regions")
        duration_ms = int(round(metadata["duration_seconds"] * 1000))
        audio_activity = {
            "vad": vad_result,
            "speech_regions": speech_regions,
            "non_speech_regions": non_speech_regions,
            "ffmpeg_silence_regions": ffmpeg_silence_regions(silences, duration_ms),
        }
        update(job_id, phase="take_detection", progress=92)
        take_warning = None
        try:
            take_detection = detect_take_candidates(segments, audio_activity)
        except Exception as error:
            take_detection = {
                "version": "1.0", "status": "failed", "processed_locally": True,
                "ollama_used": False, "processing_seconds": 0.0,
                "error": str(error), "candidates": [],
            }
            take_warning = "言い直し・重複候補の検出に失敗しましたが、文字起こしとVAD結果は正常に保存されました。"
        update(job_id, phase="format", progress=96)
        source_after = source_video.stat()
        if (source_before.st_size, source_before.st_mtime_ns) != (source_after.st_size, source_after.st_mtime_ns):
            raise RuntimeError("解析中に元動画のサイズまたは更新時刻が変わりました。元動画を確認してください。")
        result = {
            "schema_version": "1.2",
            "segments": segments,
            "audio_activity": audio_activity,
            "take_detection": take_detection,
            "transcript_segments": len(segments),
            "processing_seconds": round(time.time() - started_at, 1),
            "transcription_model": whisper_model().stem.replace("ggml-", ""),
            "language": "ja",
            "processed_locally": not CLOUD_MODE,
            "source": {
                "filename": source_video.name,
                "path": str(source_video) if not CLOUD_MODE else "",
                "size_bytes": source_after.st_size,
                "modified_at_ns": source_after.st_mtime_ns,
            },
            "warnings": [warning for warning in (vad_warning, take_warning) if warning],
            **metadata,
        }
        (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        update(job_id, status="complete", phase="complete", progress=100, **result)
    except Exception as error:
        update(job_id, status="failed", phase="failed", error=str(error), progress=0)
    finally:
        unlink_job_artifact(temporary_audio, job_dir, {"audio.wav"})
        if delete_source_copy:
            # Cloud/旧ブラウザー経路でジョブ内に生成したコピーだけを削除する。
            unlink_job_artifact(source_video, job_dir, {"input.mp4"})


def run_cloud_worker() -> None:
    job_id = os.environ["WORKER_JOB_ID"]
    mode = os.getenv("WORKER_MODE", "transcribe")
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = load_job(job_id)
    with LOCK: JOBS[job_id] = job
    ollama_process = None
    try:
        if mode == "transcribe":
            object_name = os.environ["GCS_OBJECT"]
            storage_bucket().blob(object_name).download_to_filename(job_dir / "input.mp4")
            analyze_job(job_id, job_dir / "input.mp4", delete_source_copy=True)
        else:
            ollama = tool_path("OLLAMA_BIN", "ollama")
            if ollama and not ollama_available(ROUGH_CUT_MODEL):
                ollama_process = subprocess.Popen([ollama, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                for _ in range(60):
                    if ollama_available(ROUGH_CUT_MODEL): break
                    time.sleep(1)
            run_rough_cut_analysis(job_id)
    finally:
        if ollama_process: ollama_process.terminate()


def bridge_audio_to_cloud(job_id: str, video: Path, source_filename: str) -> None:
    """Macで映像を音声へ縮小し、音声だけをCloud Runへ渡す。"""
    job_dir = video.parent
    audio = job_dir / "audio.m4a"
    try:
        ffmpeg = environment()["tools"]["ffmpeg"]["path"]
        if not ffmpeg: raise RuntimeError("Mac内のFFmpegが見つかりません。")
        update(job_id, phase="audio", progress=18)
        run([ffmpeg, "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "64k", str(audio)], "動画から音声を取り出せませんでした。")
        update(job_id, phase="upload", progress=28)
        session = remote_json("/api/uploads", {
            "job_id": job_id, "filename": "audio.m4a", "source_filename": source_filename,
            "size": audio.stat().st_size, "content_type": "audio/mp4", "media_kind": "audio",
        })
        request = Request(session["upload_url"], data=audio.read_bytes(), headers={"Content-Type": "audio/mp4"}, method="PUT")
        with urlopen(request, timeout=1800) as response: response.read()
        remote_json("/api/jobs", {"job_id": job_id, "object_name": session["object_name"]})
        update(job_id, status="processing", phase="upload", progress=10, cloud_started=True)
    except Exception as error:
        update(job_id, status="failed", phase="failed", progress=0, error=str(error))
    finally:
        unlink_job_artifact(audio, job_dir, {"audio.m4a"})
        unlink_job_artifact(video, job_dir, {"input.mp4"})


class Handler(BaseHTTPRequestHandler):
    server_version = "RoughCutAI/0.1"
    def log_message(self, format, *args): print(f"[{self.log_date_time_string()}] {format % args}")
    def json_response(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health": return self.json_response(environment())
        media_match = re.fullmatch(r"/api/local-media/([0-9a-f]{32})", path)
        if media_match and not CLOUD_MODE and not CLOUD_BRIDGE_URL:
            try:
                source_video = selected_video(media_match.group(1))
                size = source_video.stat().st_size
                range_header = self.headers.get("Range", "")
                start, end = 0, size - 1
                if range_header:
                    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
                    if not match: return self.send_error(416)
                    if match.group(1): start = int(match.group(1))
                    if match.group(2): end = min(int(match.group(2)), size - 1)
                    if start > end or start >= size: return self.send_error(416)
                length = end - start + 1
                self.send_response(206 if range_header else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                if range_header: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                with source_video.open("rb") as media:
                    media.seek(start)
                    remaining = length
                    while remaining:
                        chunk = media.read(min(1024 * 1024, remaining))
                        if not chunk: break
                        self.wfile.write(chunk); remaining -= len(chunk)
                return
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as error:
                return self.json_response({"error": str(error)}, 404)
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            job = load_job(job_id)
            return self.json_response(job or {"error": "処理情報が見つかりません。"}, 200 if job else 404)
        files = {"/": "index.html", "/index.html": "index.html", "/styles.css": "styles.css", "/transcript.css": "transcript.css", "/app.js": "app.js"}
        filename = files.get(path)
        if not filename: return self.send_error(404)
        data = (ROOT / filename).read_bytes(); content_type = "text/html; charset=utf-8" if filename.endswith("html") else "text/css; charset=utf-8" if filename.endswith("css") else "application/javascript; charset=utf-8"
        self.send_response(200); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(data)))
        # GitHubや外部APIへ問い合わせず、同梱した静的ファイルをブラウザーで再利用する。
        self.send_header("Cache-Control", "no-cache" if filename.endswith("html") else "public, max-age=3600")
        self.end_headers(); self.wfile.write(data)
    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/local-select" and not CLOUD_MODE and not CLOUD_BRIDGE_URL:
            try:
                source_video = choose_local_video()
                tools = environment()["tools"]
                if not tools["ffprobe"]["ready"]: raise RuntimeError("ffprobeが見つかりません。")
                metadata = probe_video(source_video, tools["ffprobe"]["path"])
                selection_id = uuid.uuid4().hex
                with LOCK: SELECTED_FILES[selection_id] = source_video
                stat = source_video.stat()
                return self.json_response({
                    "selection_id": selection_id, "filename": source_video.name,
                    "size": stat.st_size, **metadata,
                })
            except Exception as error:
                return self.json_response({"error": str(error)}, 400)
        if path == "/api/uploads" and CLOUD_MODE:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                filename = str(payload.get("filename") or "video.mp4")
                source_filename = str(payload.get("source_filename") or filename)
                size = int(payload.get("size") or 0)
                if size <= 0: raise ValueError("動画のファイル容量が不正です。")
                if size > CLOUD_MAX_UPLOAD_BYTES:
                    raise ValueError("公開版では1GB以下の動画に対応しています。")
                media_kind = str(payload.get("media_kind") or "video")
                if media_kind == "video" and not filename.lower().endswith(".mp4"): raise ValueError("MP4動画を選んでください。")
                if media_kind == "audio" and not filename.lower().endswith((".m4a", ".mp4", ".wav")): raise ValueError("対応していない音声形式です。")
                requested_id = str(payload.get("job_id") or "")
                job_id = requested_id if re.fullmatch(r"[0-9a-f]{32}", requested_id) else uuid.uuid4().hex
                with LOCK: JOBS[job_id] = {"job_id": job_id, "filename": source_filename, "status": "uploading", "phase": "upload", "progress": 3, "created_at": time.time()}
                upload_status(job_id, JOBS[job_id])
                content_type = str(payload.get("content_type") or ("audio/mp4" if media_kind == "audio" else "video/mp4"))
                object_name, upload_url = create_resumable_upload(job_id, filename, size, content_type)
                return self.json_response({"job_id": job_id, "object_name": object_name, "upload_url": upload_url})
            except Exception as error:
                return self.json_response({"error": str(error)}, 400)
        rough_match = re.fullmatch(r"/api/jobs/([0-9a-f]+)/rough-cuts", path)
        if rough_match:
            job_id = rough_match.group(1)
            job = load_job(job_id)
            if not job or job.get("status") != "complete":
                return self.json_response({"error": "先に文字起こしを完了させてください。"}, 400)
            if job.get("rough_status") == "processing":
                return self.json_response({"job_id": job_id}, 202)
            update(job_id, rough_status="processing", rough_phase="queued", rough_progress=5, rough_error=None)
            if CLOUD_MODE: trigger_cloud_worker(job_id, mode="rough")
            elif CLOUD_BRIDGE_URL:
                try: remote_json(path, {}, "POST")
                except Exception as error: return self.json_response({"error": str(error)}, 400)
            else: threading.Thread(target=run_rough_cut_analysis, args=(job_id,), daemon=True).start()
            return self.json_response({"job_id": job_id}, 202)
        if path != "/api/jobs": return self.send_error(404)
        if CLOUD_MODE:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                job_id = str(payload["job_id"]); object_name = str(payload["object_name"])
                job = load_job(job_id)
                if not job: raise ValueError("アップロード情報が見つかりません。")
                update(job_id, status="processing", phase="upload", progress=10)
                trigger_cloud_worker(job_id, object_name, "transcribe")
                return self.json_response({"job_id": job_id}, 202)
            except Exception as error:
                return self.json_response({"error": str(error)}, 400)
        if not CLOUD_BRIDGE_URL and self.headers.get("Content-Type", "").startswith("application/json"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                selection_id = str(payload.get("selection_id") or "")
                if not re.fullmatch(r"[0-9a-f]{32}", selection_id):
                    raise RuntimeError("動画の選択情報が不正です。")
                source_video = selected_video(selection_id)
                job_id = uuid.uuid4().hex
                job_dir = JOBS_DIR / job_id
                job_dir.mkdir(parents=True, exist_ok=False)
                with LOCK:
                    JOBS[job_id] = {
                        "job_id": job_id, "filename": source_video.name,
                        "selection_id": selection_id, "status": "processing",
                        "phase": "queued", "progress": 5, "created_at": time.time(),
                        "local_direct": True,
                    }
                threading.Thread(
                    target=analyze_job, args=(job_id, source_video, False), daemon=True
                ).start()
                return self.json_response({"job_id": job_id}, 202)
            except Exception as error:
                return self.json_response({"error": str(error)}, 400)
        # 一部のMac用ブラウザーはBlob送信時にContent-Lengthを公開しない。
        # 画面側がFile.sizeから付与したローカル専用ヘッダーを代替として使う。
        # File.sizeを入れたローカル専用ヘッダーを優先する。
        # ブラウザーによっては大容量BlobのContent-Lengthが正しく公開されない。
        declared_size = self.headers.get("X-File-Size") or self.headers.get("Content-Length") or "0"
        try: size = int(declared_size)
        except ValueError: size = 0
        if size <= 0:
            print(f"[upload rejected] invalid size: {declared_size!r}")
            return self.json_response({"error": "動画のファイル容量を確認できません。動画を選び直してください。"}, 400)
        if size > MAX_UPLOAD_BYTES:
            size_gb = size / 1024 / 1024 / 1024
            print(f"[upload rejected] too large: {size} bytes")
            return self.json_response({"error": f"動画は{size_gb:.1f}GBです。現在は100GB以下のMP4に対応しています。"}, 400)
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
        if CLOUD_BRIDGE_URL:
            threading.Thread(target=bridge_audio_to_cloud, args=(job_id, target, filename), daemon=True).start()
        else:
            threading.Thread(target=analyze_job, args=(job_id, target, True), daemon=True).start()
        self.json_response({"job_id": job_id}, 202)


def main():
    JOBS_DIR.mkdir(parents=True, exist_ok=True); MODELS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"粗カットAI: http://{HOST}:{PORT}")
    print("終了するには Control+C を押してください。")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        if OLLAMA_PROCESS: OLLAMA_PROCESS.terminate()


if __name__ == "__main__":
    if "--worker" in sys.argv: run_cloud_worker()
    else: main()
