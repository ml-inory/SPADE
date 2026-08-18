import pytest

from spade.data import CharTokenizer, SyntheticTTSDataset, ToySpeechCodec, collate_llmtts_batch, make_synthetic_texts
from spade.metrics import word_error_rate


def test_tokenizer_roundtrip():
    tok = CharTokenizer()
    text = "Hello, world! 123"
    ids = tok.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == text.lower()
    assert tok.vocab_size > 30


def test_codec_roundtrip_is_exact():
    codec = ToySpeechCodec(alphabet=CharTokenizer().alphabet)
    for text in ["the quick brown fox", "speech synthesis 123", "hello world!"]:
        for speaker in range(4):
            codes = codec.encode(text, speaker)
            decoded = codec.decode(codes, speaker)
            assert decoded == text.lower(), (text, decoded)


def test_codec_is_context_dependent():
    codec = ToySpeechCodec(alphabet=CharTokenizer().alphabet)
    a = codec.encode("cat", speaker=0)
    b = codec.encode("cat", speaker=1)
    assert a != b, "different speakers should produce different codes"
    # Same speaker, same text, different position -> different codes.
    c = codec.encode("cat", speaker=0, start_position=10)
    assert a != c


def test_dataset_and_collation():
    texts = make_synthetic_texts(8, seed=1)
    tok = CharTokenizer()
    codec = ToySpeechCodec(alphabet=tok.alphabet)
    ds = SyntheticTTSDataset(texts, codec, tok, num_speakers=2)
    sample = ds[0]
    assert sample["input_ids"].ndim == 1
    assert (sample["labels"] >= -100).all()
    assert (sample["labels"] > -100).any(), "sample must contain target codes"

    batch = collate_llmtts_batch([ds[i] for i in range(4)])
    assert batch["input_ids"].shape[0] == 4
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].max() < 512  # sanity: real code ids, not padding


def test_wer_basics():
    assert word_error_rate("hello world", "hello world") == 0.0
    assert word_error_rate("hello world", "hello") == pytest.approx(0.5)
    assert word_error_rate("hello world", "") == 1.0
    assert word_error_rate("", "") == 0.0
