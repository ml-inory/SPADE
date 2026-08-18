from spade_cosyvoice2.data_prep import collect_utterances


def test_collect_utterances(tmp_path):
    spk = tmp_path / "LibriTTS" / "dev-clean" / "100" / "121721"
    spk.mkdir(parents=True)
    (spk / "100_121721_0000.wav").write_bytes(b"RIFF")
    (spk / "100_121721_0000.normalized.txt").write_text("Hello world.")
    (spk / "100_121721_0001.wav").write_bytes(b"RIFF")
    # Files without a transcript are skipped.
    (spk / "100_121721_0002.wav").write_bytes(b"RIFF")
    utts = collect_utterances(tmp_path / "LibriTTS" / "dev-clean")
    assert len(utts) == 1
    assert utts[0]["utt"] == "100_121721_0000"
    assert utts[0]["text"] == "Hello world."
    assert utts[0]["spk"] == "100"
