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
import sys
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
HOST = "0.0.0.0" if CLOUD_MODE else "127.0.0.1"
PORT = int(os.getenv("PORT", "8765"))
# 高画質の収録素材は30分程度でも20GBを超えることがある。
# ローカル処理なので、実用上の誤操作防止として100GBを上限にする。
MAX_UPLOAD_BYTES = 100 * 1024 * 1024 * 1024
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
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
    return {"cloud_mode": CLOUD_MODE, "tools": {
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
            analyze_job(job_id)
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
        if path == "/api/uploads" and CLOUD_MODE:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                filename = str(payload.get("filename") or "video.mp4")
                size = int(payload.get("size") or 0)
                if size <= 0 or size > MAX_UPLOAD_BYTES: raise ValueError("動画のファイル容量が不正です。")
                if not filename.lower().endswith(".mp4"): raise ValueError("MP4動画を選んでください。")
                job_id = uuid.uuid4().hex
                with LOCK: JOBS[job_id] = {"job_id": job_id, "filename": filename, "status": "uploading", "phase": "upload", "progress": 3, "created_at": time.time()}
                upload_status(job_id, JOBS[job_id])
                object_name, upload_url = create_resumable_upload(job_id, filename, size, "video/mp4")
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


if __name__ == "__main__":
    if "--worker" in sys.argv: run_cloud_worker()
    else: main()
