"""voicecore CLI.

    voicecore demo                                  # from-scratch learning on synthetic proto-speech (CPU, minutes)
    voicecore init  ~/vc --preset small             # create an engine (random weights)
    voicecore train-codec ~/vc ~/recordings --steps 5000
    voicecore train-lm    ~/vc ~/recordings --steps 5000
    voicecore talk  ~/vc in.wav out.wav             # voice in -> voice out, no STT/TTS
    voicecore listen ~/vc                           # microphone loop (pip install sounddevice)
    voicecore selftrain ~/vc --rounds 3             # learn from everything heard and said
    voicecore status ~/vc | langs ~/vc | name ~/vc 0 Hindi | feedback ~/vc <id> +1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__


def _print_event(kind: str, data: dict) -> None:
    if kind == "codec_step":
        print(f"  codec step {data['step']}: loss {data['total']:.3f} (wave {data['wave']:.3f} spec {data['spec']:.3f}) {data['elapsed']}s")
    elif kind == "lm_step":
        print(f"  lm step {data['step']}: loss {data['loss']:.3f} {data['elapsed']}s")
    elif kind == "selftrain_round":
        print(f"  round {data['round']}: stage={data['stage']} " + " ".join(
            f"{k}={v}" for k, v in data.items() if k in ("codec_snr_db", "lm_loss", "dialogue_loss", "self_play_kept", "voice_clusters", "note")))
    elif kind == "selftrain_skipped":
        print(f"  skipped: {data.get('note')}")


def _engine(args: argparse.Namespace, cfg=None):
    from .engine import VoiceCore
    return VoiceCore(args.root, cfg)


def _preset(name: str):
    from .config import VoiceCoreConfig
    return {"tiny": VoiceCoreConfig.tiny, "small": VoiceCoreConfig.small, "default": VoiceCoreConfig}[name]()


def cmd_init(args) -> int:
    vc = _engine(args, _preset(args.preset))
    vc.save()
    st = vc.status()
    print(f"initialised {vc.root} ({args.preset}): codec {st['codec_params']:,} params, lm {st['lm_params']:,} params, "
          f"{st['sample_rate']} Hz, {st['frame_rate']:.0f} frames/s, device {st['device']} - all weights random")
    return 0


def cmd_train_codec(args) -> int:
    from .codec.train import CodecTrainer
    from .data import AudioCorpus

    vc = _engine(args)
    corpus = AudioCorpus(vc.sr)
    n = corpus.add_files(args.audio)
    for clip in vc.buffer.heard_clips():
        corpus.add_array(clip)
    if corpus.seconds <= 0:
        print("no audio found", file=sys.stderr)
        return 1
    print(f"{n} file(s), {corpus.seconds / 60:.1f} min of audio")
    tr = CodecTrainer(vc.codec, vc.cfg, on_event=_print_event)
    tr.train(corpus, args.steps, log_every=args.log_every)
    vc.meta["codec_steps"] = vc.meta.get("codec_steps", 0) + args.steps
    vc.save()
    print(f"codec quality: {tr.evaluate(corpus)} (saved)")
    return 0


def cmd_train_lm(args) -> int:
    from .data import AudioCorpus
    from .lm.train import LMTrainer

    vc = _engine(args)
    corpus = AudioCorpus(vc.sr)
    n = corpus.add_files(args.audio)
    for clip in vc.buffer.heard_clips():
        corpus.add_array(clip)
    if corpus.seconds <= 0:
        print("no audio found", file=sys.stderr)
        return 1
    print(f"{n} file(s), {corpus.seconds / 60:.1f} min of audio")
    tr = LMTrainer(vc.lm, vc.codec, vc.cfg, on_event=_print_event)
    loss = tr.train_continuation(corpus, args.steps, log_every=args.log_every)
    vc.meta["lm_steps"] = vc.meta.get("lm_steps", 0) + args.steps
    vc.save()
    print(f"lm loss: {loss:.3f} (saved)")
    return 0


def cmd_train_dialogue(args) -> int:
    """Learn replies from conversation recordings: turns split on silence become (heard, reply) pairs."""
    import numpy as np
    from .audio.io import load_audio
    from .audio.vad import split_on_silence
    from .lm.train import LMTrainer
    from .data import list_audio

    vc = _engine(args)
    pairs = []
    for p in list_audio(args.audio):
        wav = load_audio(p, vc.sr)
        spans = split_on_silence(wav, vc.sr, min_silence=args.min_silence)
        turns = [wav[a:b] for a, b in spans]
        pairs += [(turns[i], turns[i + 1]) for i in range(len(turns) - 1)]
    if not pairs:
        print("no turn pairs found (need recordings with pauses between speakers)", file=sys.stderr)
        return 1
    print(f"{len(pairs)} (heard, reply) pairs")
    tr = LMTrainer(vc.lm, vc.codec, vc.cfg, on_event=_print_event)
    import random
    rng = random.Random(0)
    loss = float("nan")
    started = time.time()
    for i in range(args.steps):
        batch = rng.sample(pairs, k=min(vc.cfg.train.batch_size, len(pairs)))
        limit = int(vc.cfg.max_response_seconds * vc.sr)
        loss = tr.step_exchanges([(h[:limit], r[:limit]) for h, r in batch])
        if (i + 1) % args.log_every == 0:
            print(f"  dialogue step {i + 1}: loss {loss:.3f} {time.time() - started:.0f}s")
    vc.save()
    print(f"dialogue loss: {loss:.3f} (saved)")
    return 0


def cmd_talk(args) -> int:
    vc = _engine(args)
    started = time.time()
    reply = vc.respond_file(args.input, args.output, learn=not args.no_learn)
    print(f"replied with {len(reply) / vc.sr:.2f}s of audio in {time.time() - started:.1f}s -> {args.output}")
    return 0


def cmd_roundtrip(args) -> int:
    from .audio.io import load_audio, write_wav
    from .audio.features import snr_db
    import torch

    vc = _engine(args)
    wav = load_audio(args.input, vc.sr)
    out = vc.round_trip(wav)
    write_wav(args.output, out, vc.sr)
    print(f"codec round-trip SNR {snr_db(torch.from_numpy(wav), torch.from_numpy(out)):.1f} dB -> {args.output}")
    return 0


def cmd_listen(args) -> int:
    """Microphone loop: utterance in -> reply out, learning as it goes."""
    try:
        import sounddevice as sd
    except ImportError:
        print("pip install sounddevice  (microphone support)", file=sys.stderr)
        return 1
    import numpy as np
    from .audio.vad import rms_energy

    vc = _engine(args)
    sr = vc.sr
    print(f"listening at {sr} Hz; speak, pause {args.silence}s, and it answers. Ctrl-C to stop.")
    frame = int(0.03 * sr)
    buf: list[np.ndarray] = []
    silent_for = 0.0
    thr = args.threshold
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=frame) as stream:
            while True:
                block, _ = stream.read(frame)
                x = block[:, 0].copy()
                level = float(rms_energy(x, len(x), len(x))[0])
                if level > thr:
                    buf.append(x)
                    silent_for = 0.0
                elif buf:
                    buf.append(x)
                    silent_for += len(x) / sr
                    if silent_for >= args.silence:
                        utter = np.concatenate(buf)
                        buf, silent_for = [], 0.0
                        if len(utter) / sr < 0.3:
                            continue
                        info = vc.hear(utter)
                        print(f"  heard {info['seconds']:.1f}s [{info['cluster_name']}] - replying...")
                        chunks = []
                        reply = vc.respond(utter, on_chunk=lambda c: (chunks.append(c), sd.play(c, sr)))
                        sd.wait()
                        print(f"  said {len(reply) / sr:.1f}s")
    except KeyboardInterrupt:
        print("\nstopped")
    vc.save()
    return 0


def cmd_selftrain(args) -> int:
    from .selftrain.loop import SelfTrainer
    from .data import AudioCorpus

    vc = _engine(args)
    extra = None
    if args.audio:
        extra = AudioCorpus(vc.sr)
        extra.add_files(args.audio)
    st = SelfTrainer(vc, on_event=_print_event)
    for _ in range(args.rounds):
        st.round(corpus_extra=extra)
    print(st.state)
    return 0


def cmd_status(args) -> int:
    import json
    print(json.dumps(_engine(args).status(), indent=1))
    return 0


def cmd_langs(args) -> int:
    vc = _engine(args)
    rows = vc.clusters.summary()
    if not rows:
        print("no voices/languages discovered yet - let it hear some speech")
    for r in rows:
        print(f"  {r['id']:>3}  {r['name']:<16} {r['utterances']} utterance(s)")
    return 0


def cmd_name(args) -> int:
    vc = _engine(args)
    vc.clusters.name(args.cluster, args.label)
    vc.save()
    print(f"cluster {args.cluster} is now '{args.label}'")
    return 0


def cmd_feedback(args) -> int:
    vc = _engine(args)
    ok = vc.feedback(args.exchange_id, 1 if args.value in {"+1", "1", "good", "up"} else -1)
    print("recorded" if ok else "no such exchange id")
    return 0 if ok else 1


def cmd_demo(args) -> int:
    """Everything from scratch on synthetic proto-speech in two 'languages': shows the codec learning to
    hear, the LM learning structure, replies matching the input's cluster, and a self-training round."""
    import tempfile
    import numpy as np
    import torch
    from .audio.features import snr_db
    from .codec.train import CodecTrainer
    from .config import VoiceCoreConfig
    from .data import AudioCorpus, proto_speech
    from .engine import VoiceCore
    from .lm.train import LMTrainer
    from .selftrain.loop import SelfTrainer

    root = args.root or tempfile.mkdtemp(prefix="voicecore-demo-")
    cfg = VoiceCoreConfig.tiny()
    cfg.selftrain.min_buffer_seconds = 5.0
    cfg.selftrain.codec_snr_target_db = 0.5
    cfg.selftrain.codec_steps_per_round = 20
    cfg.selftrain.lm_steps_per_round = 20
    cfg.selftrain.dialogue_steps_per_round = 10
    cfg.selftrain.self_play_samples = 2
    cfg.selftrain.candidates_per_sample = 2
    vc = VoiceCore(root, cfg)
    print(f"engine at {root}: codec {sum(p.numel() for p in vc.codec.parameters()):,} params, "
          f"lm {sum(p.numel() for p in vc.lm.parameters()):,} params - all random, nothing pretrained")

    corpus = AudioCorpus(vc.sr)
    for lang in (0, 1):
        for seed in range(4):
            corpus.add_array(proto_speech(args.seconds / 8, seed=seed, language=lang, speaker=1.0 + 0.4 * lang),
                             f"lang{lang}-{seed}")
    print(f"synthetic corpus: {corpus.seconds:.0f}s of proto-speech in 2 made-up languages")

    ct = CodecTrainer(vc.codec, cfg, on_event=_print_event)
    e0 = ct.evaluate(corpus)
    print(f"\n[1] codec, before training: SNR {e0['snr_db']:.1f} dB, log-mel distance {e0['mel_distance']:.2f}, codes in use {e0['codes_used']}/{e0['codebook_size']}")
    ct.train(corpus, args.codec_steps, log_every=max(1, args.codec_steps // 4))
    e1 = ct.evaluate(corpus)
    learned = e1["mel_distance"] < e0["mel_distance"] - 0.3 and e1["snr_db"] > e0["snr_db"]
    print(f"    codec, after {args.codec_steps} steps: SNR {e1['snr_db']:.1f} dB, log-mel distance {e1['mel_distance']:.2f}, "
          f"codes in use {e1['codes_used']}/{e1['codebook_size']}  ({'learned' if learned else 'no gain'})")

    lt = LMTrainer(vc.lm, vc.codec, cfg, on_event=_print_event)
    import random
    rng = random.Random(0)
    with torch.no_grad():
        b = corpus.sample_batch(4, 1.0, rng)
    print(f"\n[2] speech LM on codec tokens (never sees text)")
    first = lt.train_continuation(corpus, 1, log_every=1000)
    last = lt.train_continuation(corpus, args.lm_steps, log_every=max(1, args.lm_steps // 4))
    print(f"    lm loss {first:.3f} -> {last:.3f} after {args.lm_steps} steps  ({'learned' if last < first - 0.2 else 'no gain'})")

    print(f"\n[3] voice-to-voice: hear an utterance in each language, reply in audio")
    for lang in (0, 1):
        utter = proto_speech(1.0, seed=99, language=lang, speaker=1.0 + 0.4 * lang)
        info = vc.hear(utter)
        reply = vc.respond(utter, max_seconds=1.0)
        print(f"    language {lang}: heard {info['seconds']:.1f}s -> {info['cluster_name']}, replied {len(reply) / vc.sr:.2f}s of audio")
    print(f"    discovered voice/language clusters: {vc.clusters.summary()}")

    print(f"\n[4] one self-training round (codec + lm + self-play + dialogue)")
    st = SelfTrainer(vc, on_event=_print_event)
    rep = st.round(corpus_extra=corpus)
    vc.save()
    print(f"\nengine saved to {root}  (voicecore status {root})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voicecore", description="From-scratch, self-trained voice-to-voice engine.")
    p.add_argument("--version", action="version", version=f"voicecore {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="create an engine with random weights")
    s.add_argument("root"); s.add_argument("--preset", choices=["tiny", "small", "default"], default="default")
    s.set_defaults(func=cmd_init)
    for name, fn, helptext in [("train-codec", cmd_train_codec, "self-supervised codec training on raw audio"),
                               ("train-lm", cmd_train_lm, "speech LM training on raw audio (no transcripts)")]:
        s = sub.add_parser(name, help=helptext)
        s.add_argument("root"); s.add_argument("audio", nargs="+", help="wav files or directories")
        s.add_argument("--steps", type=int, default=2000); s.add_argument("--log-every", type=int, default=50)
        s.set_defaults(func=fn)
    s = sub.add_parser("train-dialogue", help="learn replies from conversation recordings (turns split on silence)")
    s.add_argument("root"); s.add_argument("audio", nargs="+"); s.add_argument("--steps", type=int, default=1000)
    s.add_argument("--min-silence", type=float, default=0.5); s.add_argument("--log-every", type=int, default=50)
    s.set_defaults(func=cmd_train_dialogue)
    s = sub.add_parser("talk", help="reply to a wav with a wav")
    s.add_argument("root"); s.add_argument("input"); s.add_argument("output"); s.add_argument("--no-learn", action="store_true")
    s.set_defaults(func=cmd_talk)
    s = sub.add_parser("roundtrip", help="encode+decode a wav through the codec (hearing quality)")
    s.add_argument("root"); s.add_argument("input"); s.add_argument("output"); s.set_defaults(func=cmd_roundtrip)
    s = sub.add_parser("listen", help="microphone conversation loop")
    s.add_argument("root"); s.add_argument("--silence", type=float, default=0.7); s.add_argument("--threshold", type=float, default=0.02)
    s.set_defaults(func=cmd_listen)
    s = sub.add_parser("selftrain", help="learn from everything heard and said")
    s.add_argument("root"); s.add_argument("--rounds", type=int, default=1); s.add_argument("--audio", nargs="*", help="extra recordings")
    s.set_defaults(func=cmd_selftrain)
    s = sub.add_parser("status"); s.add_argument("root"); s.set_defaults(func=cmd_status)
    s = sub.add_parser("langs", help="discovered voice/language clusters"); s.add_argument("root"); s.set_defaults(func=cmd_langs)
    s = sub.add_parser("name", help="label a cluster"); s.add_argument("root"); s.add_argument("cluster", type=int); s.add_argument("label")
    s.set_defaults(func=cmd_name)
    s = sub.add_parser("feedback", help="rate a reply: +1 / -1"); s.add_argument("root"); s.add_argument("exchange_id"); s.add_argument("value")
    s.set_defaults(func=cmd_feedback)
    s = sub.add_parser("demo", help="from-scratch learning on synthetic proto-speech (CPU, a few minutes)")
    s.add_argument("--root"); s.add_argument("--seconds", type=float, default=120.0)
    s.add_argument("--codec-steps", type=int, default=300); s.add_argument("--lm-steps", type=int, default=300)
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
