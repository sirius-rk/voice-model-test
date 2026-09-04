from stt_tts_benchmark.adapters import Qwen3ASRAdapter


def test_qwen3_asr_auto_language_is_detected_by_model() -> None:
    adapter = Qwen3ASRAdapter(model_id="Qwen/Qwen3-ASR-0.6B", language="Auto")

    assert adapter.language is None
    assert adapter.provider == "qwen3-asr-local"
    assert adapter.is_local is True
