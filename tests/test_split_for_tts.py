"""Standalone asserts for run._split_for_tts (no pytest dependency).

Run: python tests/test_split_for_tts.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import TTS_MAX_CHARS, _split_for_tts


def test_short_text_one_chunk():
    assert _split_for_tts("Hello world.") == ["Hello world."]


def test_empty_text_no_chunks():
    assert _split_for_tts("   ") == []


def test_splits_on_sentence_boundary_under_limit():
    s = ("A" * 3000) + ". " + ("B" * 3000) + "."
    chunks = _split_for_tts(s)
    assert len(chunks) == 2
    assert all(len(c) <= TTS_MAX_CHARS for c in chunks)


def test_hard_splits_oversized_single_sentence():
    s = "C" * (TTS_MAX_CHARS * 2 + 10)  # one giant "sentence", no boundaries
    chunks = _split_for_tts(s)
    assert len(chunks) == 3
    assert all(len(c) <= TTS_MAX_CHARS for c in chunks)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all chunker tests passed")
