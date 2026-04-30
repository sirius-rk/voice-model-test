from stt_tts_benchmark.metrics import character_error_rate, normalize_text, word_error_rate


def test_normalize_text_removes_punctuation() -> None:
    assert normalize_text("Hello, WORLD!") == "hello world"


def test_word_error_rate_exact_match() -> None:
    assert word_error_rate("hello world", "hello world") == 0


def test_character_error_rate_for_chinese() -> None:
    assert character_error_rate("你好世界", "你好") == 0.5

