import numpy as np
import torch

from voicecore.audio.features import log_mel, mel_filterbank, snr_db
from voicecore.audio.io import read_wav, resample, write_wav
from voicecore.audio.vad import split_on_silence
from voicecore.data import AudioCorpus, proto_speech


def test_wav_roundtrip(tmp_path):
    x = (np.sin(np.linspace(0, 100, 16000)) * 0.5).astype(np.float32)
    write_wav(tmp_path / "a.wav", x, 16000)
    y, sr = read_wav(tmp_path / "a.wav")
    assert sr == 16000 and y.shape == x.shape and np.abs(y - x).max() < 1e-3


def test_resample_lengths():
    x = np.random.default_rng(0).standard_normal(44100).astype(np.float32)
    assert len(resample(x, 44100, 16000)) == 16000
    assert len(resample(x, 44100, 48000)) == 48000
    assert resample(x, 16000, 16000) is not None


def test_mel_and_snr():
    fb = mel_filterbank(16000, 1024, 64)
    assert fb.shape == (64, 513) and float(fb.sum(1).min()) > 0
    x = torch.randn(2, 16000)
    assert log_mel(x, 16000).shape[1] == 64
    assert snr_db(x, x) > 80 and snr_db(x, torch.zeros_like(x)) < 1


def test_vad_splits_turns():
    sr = 16000
    t = np.arange(int(0.8 * sr)) / sr
    speech = (np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 5 * t)) * 0.5).astype(np.float32)  # syllable-like AM tone
    gap = np.zeros(int(0.6 * sr), dtype=np.float32)
    x = np.concatenate([gap, speech, gap, speech, gap])
    spans = split_on_silence(x, sr, min_silence=0.4)
    assert len(spans) == 2
    assert all(0.5 < (b - a) / sr < 1.2 for a, b in spans)


def test_proto_speech_and_corpus():
    a, b = proto_speech(1.0, seed=1, language=0), proto_speech(1.0, seed=1, language=1)
    assert a.shape == (16000,) and np.abs(a).max() <= 0.8 and not np.allclose(a, b)
    c = AudioCorpus(16000)
    c.add_array(a); c.add_array(b)
    import random
    batch = c.sample_batch(3, 0.5, random.Random(0))
    assert batch.shape == (3, 8000)
