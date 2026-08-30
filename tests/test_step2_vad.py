import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import server


class Step2VadTests(unittest.TestCase):
    def test_normalize_and_complement_keep_original_timeline(self):
        speech = server.normalize_speech_regions([(1.24, 3.5), (3.4, 4.0), (8.0, 9.25)], 10_000)
        self.assertEqual(speech, [
            {"id": 1, "start_ms": 1240, "end_ms": 4000, "duration_ms": 2760},
            {"id": 2, "start_ms": 8000, "end_ms": 9250, "duration_ms": 1250},
        ])
        self.assertEqual(server.complement_regions(speech, 10_000), [
            {"id": 1, "start_ms": 0, "end_ms": 1240, "duration_ms": 1240},
            {"id": 2, "start_ms": 4000, "end_ms": 8000, "duration_ms": 4000},
            {"id": 3, "start_ms": 9250, "end_ms": 10000, "duration_ms": 750},
        ])

    def test_missing_model_skips_vad_without_running_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "audio.wav"
            audio.write_bytes(b"wav")
            with patch("server.subprocess.run") as run_mock:
                result, warning = server.run_independent_vad(audio, 10.0, "/local/vad", root / "missing.bin")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["speech_regions"], [])
        self.assertIn("未セットアップ", warning)
        run_mock.assert_not_called()

    def test_successful_vad_output_is_parsed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio, model = root / "audio.wav", root / "ggml-silero-v6.2.0.bin"
            audio.write_bytes(b"wav"); model.write_bytes(b"model")
            output = "Detected 2 speech segments:\nSpeech segment 0: start = 124.00, end = 350.00\nSpeech segment 1: start = 800.00, end = 925.00\n"
            completed = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
            with patch("server.subprocess.run", return_value=completed):
                result, warning = server.run_independent_vad(audio, 10.0, "/local/vad", model)
        self.assertIsNone(warning)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["speech_regions"][0]["start_ms"], 1240)
        self.assertEqual(result["non_speech_regions"][1]["duration_ms"], 4500)

    def test_vad_failure_returns_warning_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio, model = root / "audio.wav", root / "model.bin"
            audio.write_bytes(b"wav"); model.write_bytes(b"model")
            completed = subprocess.CompletedProcess([], 2, stdout="", stderr="model error")
            with patch("server.subprocess.run", return_value=completed):
                result, warning = server.run_independent_vad(audio, 10.0, "/local/vad", model)
        self.assertEqual(result["status"], "failed")
        self.assertIn("文字起こし結果は正常", warning)

    def test_whisper_command_does_not_enable_vad(self):
        payload = {"transcription": [{"offsets": {"from": 0, "to": 1000}, "text": "テスト", "tokens": []}]}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            captured = []

            def fake_run(command, _message):
                captured.extend(command)
                (root / "transcript.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(server, "run", side_effect=fake_run):
                segments = server.transcribe(root / "audio.wav", root, "/local/whisper-cli", root / "whisper.bin")

        self.assertEqual(segments[0]["text"], "テスト")
        self.assertNotIn("--vad", captured)
        self.assertNotIn("-vm", captured)


if __name__ == "__main__":
    unittest.main()
