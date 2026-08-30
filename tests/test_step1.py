import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import server


class Step1SafetyTests(unittest.TestCase):
    def test_cloud_limit_is_1gb_without_changing_local_limit(self):
        self.assertEqual(server.CLOUD_MAX_UPLOAD_BYTES, 1024 * 1024 * 1024)
        self.assertEqual(server.MAX_UPLOAD_BYTES, 100 * 1024 * 1024 * 1024)

    def test_whisper_tokens_are_preserved_without_words(self):
        payload = {
            "transcription": [{
                "timestamps": {"from": "00:00:00,000", "to": "00:00:02,000"},
                "offsets": {"from": 0, "to": 2000},
                "text": "イエーイ",
                "tokens": [{
                    "text": "イ", "offsets": {"from": 20, "to": 320},
                    "id": 8040, "p": 0.147854, "t_dtw": -1,
                }],
            }]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "transcript.json"
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            segments = server.parse_whisper_json(source)

        self.assertEqual(segments[0]["start_ms"], 0)
        self.assertEqual(segments[0]["end_ms"], 2000)
        self.assertNotIn("words", segments[0])
        self.assertEqual(segments[0]["tokens"][0], {
            "text": "イ", "start_ms": 20, "end_ms": 320,
            "token_id": 8040, "probability": 0.147854, "t_dtw": -1,
        })

    def test_deletion_guard_rejects_source_outside_job_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_dir = root / "job"
            job_dir.mkdir()
            source_video = root / "original.mp4"
            source_video.write_bytes(b"original-video")

            with self.assertRaises(RuntimeError):
                server.unlink_job_artifact(source_video, job_dir, {"input.mp4"})

            self.assertEqual(source_video.read_bytes(), b"original-video")

    def test_deletion_guard_allows_only_named_job_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job"
            job_dir.mkdir()
            audio = job_dir / "audio.wav"
            audio.write_bytes(b"temporary")

            server.unlink_job_artifact(audio, job_dir, {"audio.wav"})

            self.assertFalse(audio.exists())

    def test_direct_analysis_keeps_original_and_creates_no_video_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jobs_dir = root / "jobs"
            jobs_dir.mkdir()
            source_video = root / "original.mp4"
            source_video.write_bytes(b"read-only-source-content")
            before = source_video.stat()
            job_id = "a" * 32
            (jobs_dir / job_id).mkdir()
            server.JOBS[job_id] = {"job_id": job_id, "filename": source_video.name}

            fake_tools = {name: {"ready": True, "path": name} for name in ("ffmpeg", "ffprobe", "whisper", "whisper_model")}

            def fake_extract(_source, audio, _ffmpeg):
                audio.write_bytes(b"wav")
                return [{"start": 0.4, "end": 0.9}]

            segment = {"id": 1, "start": 0.02, "end": 0.32, "start_ms": 20, "end_ms": 320, "text": "テスト", "tokens": []}
            with patch.object(server, "JOBS_DIR", jobs_dir), \
                 patch.object(server, "environment", return_value={"tools": fake_tools}), \
                 patch.object(server, "probe_video", return_value={"duration_seconds": 1.0, "source_fps": 30.0, "source_fps_ratio": "30/1"}), \
                 patch.object(server, "extract_audio_and_silence", side_effect=fake_extract), \
                 patch.object(server, "transcribe", return_value=[segment]), \
                 patch.object(server, "whisper_model", return_value=Path("ggml-test.bin")), \
                 patch.object(server, "upload_status", return_value=None):
                server.analyze_job(job_id, source_video, delete_source_copy=False)

            after = source_video.stat()
            self.assertEqual(source_video.read_bytes(), b"read-only-source-content")
            self.assertEqual((before.st_size, before.st_mtime_ns), (after.st_size, after.st_mtime_ns))
            self.assertFalse((jobs_dir / job_id / "input.mp4").exists())
            self.assertFalse((jobs_dir / job_id / "audio.wav").exists())
            result = json.loads((jobs_dir / job_id / "result.json").read_text(encoding="utf-8"))
            self.assertTrue(result["processed_locally"])
            self.assertEqual(result["source"]["path"], str(source_video.resolve()))
            self.assertEqual(result["segments"], [segment])
            self.assertEqual(result["audio_activity"]["vad"]["status"], "unavailable")
            self.assertEqual(result["audio_activity"]["ffmpeg_silence_regions"], [
                {"id": 1, "start_ms": 400, "end_ms": 900, "duration_ms": 500}
            ])


if __name__ == "__main__":
    unittest.main()
