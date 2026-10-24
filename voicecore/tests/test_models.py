
import numpy as np
import pytest
import torch

from voicecore.codec.losses import codec_loss
from voicecore.codec.model import AudioCodec
from voicecore.codec.train import CodecTrainer
from voicecore.config import VoiceCoreConfig
from voicecore.data import AudioCorpus, proto_speech
from voicecore.lm.generate import generate
from voicecore.lm.model import SpeechLM
from voicecore.lm.train import LMTrainer, build_exchange, pad_batch, tokenize


@pytest.fixture
def cfg():
    torch.manual_seed(0)
    return VoiceCoreConfig.tiny()


def test_codec_shapes_and_codes(cfg):
    codec = AudioCodec(cfg.codec)
    wav = torch.randn(2, 16000) * 0.1
    recon, codes, commit = codec(wav)
    assert recon.shape == (2, 16000)
    assert codes.shape == (2, cfg.codec.n_codebooks, 50) and int(codes.max()) < cfg.codec.codebook_size
    assert commit.item() >= 0
    assert codec.decode(codes).shape[-1] == 50 * codec.hop
    assert codec.embed(wav).shape == (2, cfg.codec.latent_dim)
    assert codec.fingerprint(wav).shape == (2, 2 * cfg.codec.latent_dim)
    assert codec.decode(codes).shape == (2, 16000)
    loss, parts = codec_loss(wav, recon, commit, 16000)
    loss.backward()
    assert parts["total"] > 0


def test_codec_learns_to_reconstruct(cfg):
    corpus = AudioCorpus(16000)
    for s in range(3):
        corpus.add_array(proto_speech(3.0, seed=s))
    codec = AudioCodec(cfg.codec)
    tr = CodecTrainer(codec, cfg)
    before = tr.evaluate(corpus)
    tr.train(corpus, 200, log_every=1000)
    after = tr.evaluate(corpus)
    assert after["mel_distance"] < before["mel_distance"] - 0.5, (before, after)   # timbre is being learned
    assert after["snr_db"] > before["snr_db"] + 1.0, (before, after)                 # and the waveform itself
    # codebook usage plateaus at 2-4 codes for the first ~150 steps and then diversifies; no total collapse
    assert min(after["codes_used"]) >= 2 and max(after["codes_used"]) >= 8, after


def test_dead_codes_get_reseeded(cfg):
    cfg.codec.dead_code_steps = 3
    codec = AudioCodec(cfg.codec)
    codec.train()
    vq = codec.rvq.layers[0]
    codec(torch.randn(2, 8000) * 0.1)
    assert bool(vq.initialised)                       # data-dependent init happened on the first forward
    seeded = vq.embed.clone()
    # a batch that always maps to the same few codes leaves the rest unused -> re-seeded after 3 steps
    same = torch.sin(torch.linspace(0, 100, 8000)).repeat(2, 1) * 0.1
    for _ in range(5):
        codec(same)
    assert float(vq.usage.max()) < cfg.codec.dead_code_steps
    assert not torch.allclose(vq.embed, seeded)       # unused codes were moved
    # init never re-runs, even on a much larger batch
    before = vq.embed.clone()
    codec(torch.randn(8, 16000) * 0.1)
    assert int(vq.initialised) == 1 and (vq.embed - before).abs().max() < 1.0


def test_inference_helpers_do_not_touch_codebooks(cfg):
    codec = AudioCodec(cfg.codec)
    codec.train()                                      # as left behind by CodecTrainer.step
    codec(torch.randn(2, 8000) * 0.1)
    step_before = int(codec.rvq.layers[0].step)
    embed_before = codec.rvq.layers[0].embed.clone()
    wav = torch.randn(1, 8000) * 0.1
    a = codec.encode(wav); codec.decode(a); codec.embed(wav); codec.fingerprint(wav)
    b = codec.encode(wav)
    assert torch.equal(a, b) and int(codec.rvq.layers[0].step) == step_before
    assert torch.equal(codec.rvq.layers[0].embed, embed_before) and codec.training
    assert not torch.isnan(codec.fingerprint(torch.randn(1, 320))).any()   # single frame


def test_short_inputs_do_not_crash():
    from voicecore.audio.features import log_mel
    from voicecore.codec.losses import mel_distance
    assert log_mel(torch.randn(1, 300), 16000).shape[0] == 1
    assert mel_distance(torch.randn(1, 400), torch.randn(1, 400), 16000) >= 0


def test_lm_forward_loss_generate(cfg):
    codec = AudioCodec(cfg.codec)
    lm = SpeechLM(cfg.lm, cfg.codec)
    wav = torch.randn(2, 16000) * 0.1
    codes, style = tokenize(codec, wav)
    logits = lm(codes, style)
    assert logits.shape == (2, cfg.codec.n_codebooks, 51, lm.special.vocab)
    loss = lm.loss(codes, style)
    loss.backward()
    assert 0 < loss.item() < 10
    out = generate(lm, codes[:, :, :5], style, max_new_frames=12, top_k=8)
    assert out.shape[:2] == (2, cfg.codec.n_codebooks) and out.shape[-1] <= 12
    assert int(out.max()) < cfg.codec.codebook_size
    assert len(generate.last_lengths) == 2 and all(0 <= n <= out.shape[-1] for n in generate.last_lengths)
    chunks = []
    generate(lm, codes[:1, :, :5], style[:1], max_new_frames=10, chunk_callback=chunks.append, chunk_frames=4)
    assert sum(c.shape[-1] for c in chunks) <= 10 and len(chunks) >= 2


def test_exchange_format_and_padding(cfg):
    lm = SpeechLM(cfg.lm, cfg.codec)
    sp = lm.special
    heard = torch.zeros(1, 2, 7, dtype=torch.long)
    reply = torch.ones(1, 2, 5, dtype=torch.long)
    seq, mask = build_exchange(lm, heard, reply)
    assert seq.shape == (1, 2, 1 + 7 + 1 + 5 + 1)
    assert int(seq[0, 0, 0]) == sp.bos and int(seq[0, 0, 8]) == sp.sep and int(seq[0, 0, -1]) == sp.eos
    assert mask.sum() == 6 and bool(mask[0, 9]) and not bool(mask[0, 8])
    codes, m = pad_batch(lm, [(seq, mask), (seq[:, :, :10], mask[:, :10])])
    assert codes.shape == (2, 2, 15) and int(codes[1, 0, -1]) == sp.pad and not bool(m[1, -1])


def test_lm_learns_structure(cfg):
    corpus = AudioCorpus(16000)
    for s in range(3):
        corpus.add_array(proto_speech(3.0, seed=s))
    codec = AudioCodec(cfg.codec)
    CodecTrainer(codec, cfg).train(corpus, 30, log_every=1000)
    lm = SpeechLM(cfg.lm, cfg.codec)
    tr = LMTrainer(lm, codec, cfg)
    first = tr.train_continuation(corpus, 1, log_every=1000)
    last = tr.train_continuation(corpus, 60, log_every=1000)
    assert last < first - 0.3, (first, last)
    # exchange training runs and scores
    h, r = proto_speech(0.6, seed=7), proto_speech(0.6, seed=8)
    l1 = tr.step_exchanges([(h, r)])
    l2 = tr.step_exchanges([(h, r)], weights=[2.0])
    assert l1 > 0 and l2 > 0
    assert isinstance(tr.exchange_logprob(h, r), float)
    # empty replies are skipped, never crash
    import math
    assert math.isnan(tr.step_exchanges([(h, np.zeros(0, dtype=np.float32))]))
    assert tr.exchange_logprob(h, np.zeros(0, dtype=np.float32)) == -1e9
    # a heard utterance longer than the context is tail-truncated so the reply is still scored
    long_h = proto_speech(8.0, seed=9)                       # 400 frames > max_frames 256
    lp_a = tr.exchange_logprob(long_h, r)
    lp_b = tr.exchange_logprob(long_h, proto_speech(0.3, seed=10))   # different length -> different token sequence
    assert math.isfinite(lp_a) and math.isfinite(lp_b) and lp_a != lp_b
