import json

import numpy as np
import pytest

from voicecore.config import VoiceCoreConfig
from voicecore.data import AudioCorpus, proto_speech
from voicecore.engine import VoiceCore
from voicecore.engine import FINGERPRINT_DIM
from voicecore.lang.cluster import VoiceClusters
from voicecore.selftrain.buffer import ExperienceBuffer
from voicecore.selftrain.loop import SelfTrainer


def _engine(tmp_path):
    cfg = VoiceCoreConfig.tiny()
    cfg.max_response_seconds = 0.5
    return VoiceCore(tmp_path / "vc", cfg)


def test_hear_respond_stream_save_load(tmp_path):
    vc = _engine(tmp_path)
    wav = proto_speech(1.0, seed=1)
    info = vc.hear(wav)
    assert info["cluster"] == 0 and info["seconds"] == 1.0
    reply = vc.respond(wav)
    assert reply.dtype == np.float32 and 0 < len(reply) <= int(0.5 * vc.sr) + vc.codec.hop
    chunks = list(vc.stream(wav, learn=False))
    assert chunks and all(c.ndim == 1 for c in chunks)
    # 44.1 kHz input is resampled
    assert vc.respond(proto_speech(1.0, sr=44100), 44100, learn=False).dtype == np.float32
    rt = vc.round_trip(wav)
    assert rt.shape == wav.shape
    st = vc.status()
    assert st["experience"]["exchanges"] == 1 and st["experience"]["heard_utterances"] == 1
    vc.save()
    vc2 = VoiceCore(tmp_path / "vc")           # config + weights + clusters reload
    assert vc2.cfg.codec.codebook_size == 64 and vc2.clusters.counts == vc.clusters.counts
    assert np.allclose(vc2.round_trip(wav), rt)


def test_relative_root_and_moved_engine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = VoiceCoreConfig.tiny(); cfg.max_response_seconds = 0.5
    vc = VoiceCore("rel_engine", cfg)                      # relative root
    vc.hear(proto_speech(0.5, seed=1))
    vc.respond(proto_speech(0.5, seed=2))
    rows = vc.buffer.rows
    assert all(not str(r.get("path", r.get("heard"))).startswith("/") for r in rows)   # stored relative
    (tmp_path / "rel_engine").rename(tmp_path / "moved")
    monkeypatch.chdir(tmp_path / "moved")
    vc2 = VoiceCore(tmp_path / "moved")
    assert len(vc2.buffer.heard_clips()) == 2 and vc2.buffer.heard_seconds() > 0.9


def test_init_refuses_mismatched_config(tmp_path):
    vc = _engine(tmp_path); vc.save()
    big = VoiceCoreConfig.tiny(); big.codec.latent_dim = 48
    with pytest.raises(RuntimeError):
        VoiceCore(tmp_path / "vc", big)
    assert VoiceCore(tmp_path / "vc").cfg.codec.latent_dim == 32      # config.json was not overwritten


def test_stereo_input_and_empty_reply(tmp_path):
    vc = _engine(tmp_path)
    stereo = np.stack([proto_speech(0.5, seed=1)] * 2, axis=1)
    assert vc.hear(stereo)["seconds"] == 0.5
    assert vc.respond(stereo, learn=False).ndim == 1
    # an empty reply is not stored as an exchange, the utterance is still remembered as heard
    info = vc.converse(proto_speech(0.5, seed=3), max_seconds=0.0)
    assert info["reply_seconds"] == 0 and info["exchange_id"] is None
    assert vc.buffer.stats()["exchanges"] == 0 and vc.buffer.stats()["heard_utterances"] == 2


def test_self_play_does_not_pollute_heard_pool(tmp_path):
    buf = ExperienceBuffer(tmp_path / "exp", 16000)
    buf.add_heard(proto_speech(0.5, seed=1))
    buf.add_exchange(proto_speech(0.5, seed=1), proto_speech(0.4, seed=2), origin="self-play")
    buf.add_exchange(proto_speech(0.5, seed=4), proto_speech(0.4, seed=2), origin="live")
    assert len(buf.heard_clips()) == 2 and len(buf.exchanges()) == 2
    assert abs(buf.heard_seconds() - 1.0) < 0.01


def test_respond_file(tmp_path):
    from voicecore.audio.io import write_wav, read_wav
    vc = _engine(tmp_path)
    write_wav(tmp_path / "in.wav", proto_speech(0.8, seed=2), 16000)
    vc.respond_file(tmp_path / "in.wav", tmp_path / "out.wav")
    y, sr = read_wav(tmp_path / "out.wav")
    assert sr == 16000 and len(y) > 0


def test_experience_buffer_and_feedback(tmp_path):
    buf = ExperienceBuffer(tmp_path / "exp", 16000)
    buf.add_heard(proto_speech(0.5, seed=1), cluster=0)
    row = buf.add_exchange(proto_speech(0.5, seed=2), proto_speech(0.4, seed=3), score=0.1)
    assert buf.feedback(row["id"], -1) and not buf.feedback("nope", 1)
    assert buf.exchanges() == []                     # negative feedback excluded
    assert buf.exchanges(exclude_negative=False)[0]["feedback"] == -1
    buf2 = ExperienceBuffer(tmp_path / "exp", 16000)  # persisted
    assert buf2.stats()["negative_feedback"] == 1 and len(buf2.heard_clips()) == 2
    assert abs(buf2.heard_seconds() - 1.0) < 0.01    # heard side of the exchange counts, the reply does not


def test_voice_clusters_discover_two_languages(tmp_path):
    vc = _engine(tmp_path)
    for seed in range(4):
        vc.hear(proto_speech(3.0, seed=seed, language=0, speaker=1.0))
    for seed in range(4):
        vc.hear(proto_speech(3.0, seed=seed, language=1, speaker=1.8))
    ids_a = {vc.clusters.identify(vc.fingerprint(proto_speech(3.0, seed=9, language=0, speaker=1.0)))}
    ids_b = {vc.clusters.identify(vc.fingerprint(proto_speech(3.0, seed=9, language=1, speaker=1.8)))}
    assert len(vc.clusters.counts) >= 2 and ids_a != ids_b
    vc.clusters.name(0, "lang-A")
    assert vc.clusters.label(0) == "lang-A" and vc.clusters.label(None) == "unknown"
    vc.save()
    loaded = VoiceClusters.load(tmp_path / "vc" / "clusters.json")
    assert loaded.names[0] == "lang-A" and loaded.dim == FINGERPRINT_DIM and loaded._n_in == vc.clusters._n_in


def test_self_training_round(tmp_path):
    vc = _engine(tmp_path)
    sc = vc.cfg.selftrain
    sc.min_buffer_seconds = 2.0
    sc.codec_snr_target_db = -100.0     # skip the curriculum gate for speed
    sc.codec_steps_per_round = 5
    sc.lm_steps_per_round = 5
    sc.dialogue_steps_per_round = 3
    sc.self_play_samples = 2
    sc.candidates_per_sample = 2
    for s in range(3):
        vc.hear(proto_speech(1.0, seed=s))
    events = []
    st = SelfTrainer(vc, on_event=lambda k, d: events.append(k))
    rep = st.round()
    assert rep["stage"] == "dialogue" and rep["self_play_kept"] >= 1 and "dialogue_loss" in rep
    assert st.state["rounds"] == 1 and (tmp_path / "vc" / "selftrain_state.json").exists()
    assert vc.buffer.stats()["exchanges"] >= 1
    assert "selftrain_round" in events and "lm_step" in events
    # a second trainer resumes state
    assert SelfTrainer(vc).state["rounds"] == 1


def test_self_training_waits_for_audio(tmp_path):
    vc = _engine(tmp_path)
    rep = SelfTrainer(vc).round()
    assert "need" in rep["note"]


def test_curriculum_gate(tmp_path):
    vc = _engine(tmp_path)
    sc = vc.cfg.selftrain
    sc.min_buffer_seconds = 1.0
    sc.codec_snr_target_db = 1000.0
    sc.codec_steps_per_round = 2
    vc.hear(proto_speech(2.0, seed=1))
    rep = SelfTrainer(vc).round()
    assert rep["stage"] == "codec" and "lm_loss" not in rep
