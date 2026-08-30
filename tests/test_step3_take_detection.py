import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import server


def segment(identifier, start, end, text):
    return {
        "id": identifier, "start": start, "end": end,
        "start_ms": int(start * 1000), "end_ms": int(end * 1000),
        "text": text, "tokens": [],
    }


ACTIVITY = {
    "speech_regions": [
        {"id": 1, "start_ms": 0, "end_ms": 5000, "duration_ms": 5000},
        {"id": 2, "start_ms": 5500, "end_ms": 12000, "duration_ms": 6500},
    ],
    "non_speech_regions": [], "ffmpeg_silence_regions": [],
}


class Step3TakeDetectionTests(unittest.TestCase):
    def test_explicit_retry_signal_is_strong_with_completed_retake(self):
        segments = [
            segment(1, 0, 3, "最近の家は断熱性能が高いです"),
            segment(2, 3.2, 4.5, "もう一回お願いします"),
            segment(3, 5.5, 9, "最近の家は断熱性能が高いです"),
        ]
        with patch.object(server, "ollama_json", side_effect=AssertionError("Ollama must not be called")):
            result = server.detect_take_candidates(segments, ACTIVITY)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["category"], "explicit_ng")
        self.assertEqual(candidate["decision"], "strong")
        self.assertEqual(candidate["replacement_segment_ids"], [3])
        self.assertFalse(result["ollama_used"])

    def test_false_start_is_detected_but_text_addition_stays_review(self):
        segments = [
            segment(1, 0, 2, "断熱性能が高い家ではその"),
            segment(2, 3, 7, "断熱性能が高い家ではエアコンを弱く使えます"),
        ]
        candidate = server.detect_take_candidates(segments, ACTIVITY)["candidates"][0]
        self.assertEqual(candidate["category"], "false_start")
        self.assertEqual(candidate["decision"], "review")
        self.assertIn("substantial_text_addition", candidate["signals"])
        self.assertEqual(candidate["detected_end_ms"], 2000)
        # 遠いVAD境界へ広げず、置き換え発言を巻き込まない。
        self.assertEqual(candidate["suggested_cut_end_ms"], 2000)

    def test_nearly_identical_retake_is_strong(self):
        segments = [
            segment(1, 0, 3, "平屋は生活動線を短くできます"),
            segment(2, 5, 8, "平屋は生活動線を短くできます"),
        ]
        candidate = server.detect_take_candidates(segments, ACTIVITY)["candidates"][0]
        self.assertEqual(candidate["category"], "near_duplicate")
        self.assertEqual(candidate["decision"], "strong")
        self.assertEqual(candidate["metrics"]["textual_addition_ratio"], 0.0)

    def test_added_number_and_supplement_are_not_strong(self):
        segments = [
            segment(1, 0, 3, "室温を下げると快適です"),
            segment(2, 5, 9, "室温を二十六度に下げると快適です"),
        ]
        result = server.detect_take_candidates(segments, ACTIVITY)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["decision"], "review")
        self.assertIn("new_distinctive_terms", candidate["signals"])
        self.assertGreater(candidate["metrics"]["textual_addition_ratio"], 0)
        self.assertIn("意味的重要度ではありません", result["metrics_note"])

    def test_short_acknowledgements_are_not_candidates(self):
        segments = [segment(1, 0, 1, "はい"), segment(2, 2, 3, "そうですね")]
        self.assertEqual(server.detect_take_candidates(segments, ACTIVITY)["candidates"], [])

    def test_raw_and_suggested_boundaries_are_both_preserved(self):
        segments = [
            segment(1, .2, 2.7, "平屋は移動しやすいです"),
            segment(2, 5.5, 8, "平屋は移動しやすいです"),
        ]
        candidate = server.detect_take_candidates(segments, ACTIVITY)["candidates"][0]
        self.assertEqual((candidate["detected_start_ms"], candidate["detected_end_ms"]), (200, 2700))
        self.assertEqual(candidate["suggested_cut_start_ms"], 0)
        self.assertEqual(candidate["suggested_cut_end_ms"], 2700)
        self.assertEqual(candidate["boundary_source"], "vad")

    def test_existing_segments_tokens_and_audio_activity_are_not_mutated(self):
        segments = [
            segment(1, 0, 3, "平屋は生活動線を短くできます"),
            segment(2, 5, 8, "平屋は生活動線を短くできます"),
        ]
        segments[0]["tokens"] = [{"text": "平屋", "start_ms": 0, "end_ms": 500}]
        activity = copy.deepcopy(ACTIVITY)
        before_segments = copy.deepcopy(segments)
        before_activity = copy.deepcopy(activity)
        server.detect_take_candidates(segments, activity)
        self.assertEqual(segments, before_segments)
        self.assertEqual(activity, before_activity)

    def test_detection_failure_does_not_destroy_step2_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir); jobs = root / "jobs"; jobs.mkdir()
            source = root / "source.mp4"; source.write_bytes(b"source")
            job_id = "e" * 32; (jobs / job_id).mkdir()
            original_segment = segment(1, 0, 1, "文字起こし結果")
            server.JOBS[job_id] = {"job_id": job_id, "filename": source.name, "status": "processing"}
            tools = {name: {"ready": True, "path": name} for name in ("ffmpeg", "ffprobe", "whisper", "whisper_model")}
            tools["vad"] = {"ready": True, "path": "vad"}

            def fake_extract(_source, audio, _ffmpeg): audio.write_bytes(b"wav"); return []
            vad_result = ({"status": "complete", "speech_regions": [], "non_speech_regions": []}, None)
            with patch.object(server, "JOBS_DIR", jobs), \
                 patch.object(server, "environment", return_value={"tools": tools}), \
                 patch.object(server, "probe_video", return_value={"duration_seconds": 1.0, "source_fps": 30.0, "source_fps_ratio": "30/1"}), \
                 patch.object(server, "extract_audio_and_silence", side_effect=fake_extract), \
                 patch.object(server, "transcribe", return_value=[original_segment]), \
                 patch.object(server, "run_independent_vad", return_value=vad_result), \
                 patch.object(server, "detect_take_candidates", side_effect=RuntimeError("test failure")), \
                 patch.object(server, "whisper_model", return_value=Path("model.bin")), \
                 patch.object(server, "upload_status", return_value=None):
                server.analyze_job(job_id, source, False)

            result = json.loads((jobs / job_id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(server.JOBS[job_id]["status"], "complete")
            self.assertEqual(result["segments"], [original_segment])
            self.assertEqual(result["take_detection"]["status"], "failed")
            self.assertIn("正常に保存", result["warnings"][0] if result["warnings"] else "")


if __name__ == "__main__":
    unittest.main()
