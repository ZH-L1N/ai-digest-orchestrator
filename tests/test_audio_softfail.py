"""Soft-fail tests for the audio broadcast path (no network, no pytest).

Run: python -m unittest tests.test_audio_softfail -v
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run


class AudioSoftFail(unittest.TestCase):
    def test_missing_secret_disables_audio_softly(self):
        # A missing audio secret -> soft error, no synthesis attempted.
        result, is_error = run.handle_send_audio_broadcast(
            None, "xoxb-tok", "C123", "2026-06-07", {"script": "hello"}
        )
        self.assertTrue(is_error)
        self.assertFalse(result["sent"])
        self.assertIn("audio disabled", result["error"])
        self.assertIn("OPENAI_API_KEY", result["error"])

    def test_empty_script_is_soft_error(self):
        result, is_error = run.handle_send_audio_broadcast(
            "sk", "xoxb-tok", "C123", "2026-06-07", {"script": "   "}
        )
        self.assertTrue(is_error)
        self.assertFalse(result["sent"])

    def test_tts_http_error_propagates_soft(self):
        with mock.patch.object(run, "_tts_one", return_value=(False, None, "http 401")):
            result, is_error = run.handle_send_audio_broadcast(
                "sk", "xoxb-tok", "C123", "2026-06-07", {"script": "real script."}
            )
        self.assertTrue(is_error)
        self.assertIn("tts failed", result["error"])
        self.assertIn("401", result["error"])

    def test_slack_ok_false_is_soft_failure(self):
        with mock.patch.object(
            run, "synthesize_tts", return_value=(True, b"ID3xxxx", "1 chunk(s), concat=single")
        ), mock.patch.object(
            run, "_slack_api_post",
            side_effect=RuntimeError("slack files.getUploadURLExternal error: not_in_channel"),
        ):
            result, is_error = run.handle_send_audio_broadcast(
                "sk", "xoxb-tok", "C123", "2026-06-07", {"script": "real script."}
            )
        self.assertTrue(is_error)
        self.assertIn("slack upload failed", result["error"])

    def test_single_chunk_concat_passthrough(self):
        audio, method = run._concat_mp3([b"ABC"])
        self.assertEqual(audio, b"ABC")
        self.assertEqual(method, "single")

    def test_unexpected_exception_is_soft(self):
        # An unexpected raise anywhere in synth/upload must become a soft error,
        # never propagate out and crash the run (the audio soft-fail contract).
        with mock.patch.object(run, "synthesize_tts", side_effect=OSError("disk full")):
            result, is_error = run.handle_send_audio_broadcast(
                "sk", "xoxb-tok", "C123", "2026-06-07", {"script": "real script."}
            )
        self.assertTrue(is_error)
        self.assertFalse(result["sent"])
        self.assertIn("audio error", result["error"])

    def test_concat_byte_join_fallback_when_no_ffmpeg(self):
        # With ffmpeg "absent", multi-chunk concat must fall back to byte-join,
        # never raise.
        with mock.patch.object(run.shutil, "which", return_value=None):
            audio, method = run._concat_mp3([b"AAA", b"BBB"])
        self.assertEqual(audio, b"AAABBB")
        self.assertEqual(method, "byte-join")


if __name__ == "__main__":
    unittest.main(verbosity=2)
