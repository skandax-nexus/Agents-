# VoiceCore

A **from-scratch, self-trained, voice-to-voice core engine**.

- **Voice in, voice out.** No speech-to-text, no text-to-speech, no text anywhere. Sound is
  encoded into discrete audio tokens by a neural codec, a speech language model continues the
  token stream with a reply, and the codec turns the reply back into a waveform. Replies are
  streamed chunk by chunk as they are generated.
- **No pretrained models.** Every weight starts random. Nothing is downloaded. The engine
  learns only from the audio it hears, from its own exchanges, and from your thumbs-up/down.
- **Multilingual by construction.** It models sound, not any particular language, so every
  language is handled the same way. It discovers the voices and languages it hears as
  clusters on its own, and replies in the voice and language it was spoken to in (the reply is
  conditioned on the input utterance's style embedding).
- **Learns on itself.** A self-training loop keeps improving the codec and the speech model
  from everything heard, practises replies by self-play with self-scoring, and weights in your
  feedback. A curriculum gate stops it from training on nonsense.

```
                  ┌──────────── self-training (codec + LM + self-play + feedback) ◄─────────┐
                  ▼                                                                         │
 mic/wav ─► Audio codec ─► tokens ─► Speech LM ─► reply tokens ─► Audio codec ─► wav ─► experience
            (encoder+RVQ)   20 ms     (transformer,   streamed      (decoder)             buffer
                                      style-conditioned)
                  └─► acoustic fingerprint ─► voice/language clusters (unsupervised)
```

## Honest expectations

This is the *core engine* and training machinery, verified end to end. Fluent conversation
from scratch is a matter of data and compute, not of code: the same architecture family
(neural codec + speech LM) reaches intelligible speech only after hundreds to thousands of
hours of audio and GPU-days of training. On a laptop with a few hours of recordings, expect the
codec to reproduce voices well and the speech model to produce voice-like babble in the right
voice and language, improving with more hearing. See `docs/ARCHITECTURE.md` for what each
stage needs.

## Quick start

```bash
pip install -e ".[dev]"            # numpy + torch (+ sounddevice for the microphone: .[mic])
pytest -q                          # 18 tests, CPU, under a minute

voicecore demo                     # everything from scratch on synthetic proto-speech (CPU, ~2 min)
```

The demo shows, with no pretrained anything: the codec learning to hear (reconstruction SNR
rising), the speech model learning the structure of the sound it hears (loss falling), two
made-up languages being discovered as separate clusters, voice-to-voice replies, and a
self-training round.

With your own recordings (any language, 16 kHz mono wav is best; anything is resampled):

```bash
voicecore init ~/vc --preset small
voicecore train-codec ~/vc ~/recordings --steps 20000     # self-supervised: learn to hear
voicecore train-lm    ~/vc ~/recordings --steps 20000     # self-supervised: learn the structure of speech
voicecore train-dialogue ~/vc ~/conversations --steps 5000  # optional: turns split on silence -> (heard, reply)
voicecore talk ~/vc question.wav answer.wav                # voice in -> voice out
voicecore listen ~/vc                                     # microphone loop; it learns while talking
voicecore selftrain ~/vc --rounds 5                       # learn from everything heard and said
voicecore langs ~/vc ; voicecore name ~/vc 0 Hindi        # discovered voices/languages, label one
voicecore feedback ~/vc <exchange-id> +1                  # teach it which replies were good
voicecore status ~/vc
```

## Python API

```python
from voicecore import VoiceCore, VoiceCoreConfig
from voicecore.audio.io import load_audio, write_wav

vc = VoiceCore("~/vc", VoiceCoreConfig.small())     # random weights the first time, reloaded after
wav = load_audio("hello.wav", vc.sr)
print(vc.hear(wav))                                  # {'cluster': 0, 'cluster_name': 'cluster-0', ...}
reply = vc.respond(wav, on_chunk=lambda chunk: ...)  # streamed audio chunks, then the full reply
write_wav("reply.wav", reply, vc.sr)

from voicecore.selftrain import SelfTrainer
SelfTrainer(vc).round()                              # learn from what it has heard and said
vc.save()
```

## Layout

`voicecore/audio` (wav I/O, resampling, mel features, VAD) - `voicecore/codec` (encoder, residual
VQ, decoder, losses, trainer) - `voicecore/lm` (speech transformer, exchange format, generation,
trainer) - `voicecore/lang` (unsupervised voice/language clusters) - `voicecore/selftrain`
(experience buffer, self-training rounds) - `voicecore/engine.py` - `voicecore/cli.py` -
`voicecore/data.py` (corpus + synthetic proto-speech).
