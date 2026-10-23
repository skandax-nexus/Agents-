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

The demo shows, with no pretrained anything: the codec learning to hear (log-mel distance
falling, waveform SNR rising, codebook usage growing after an initial plateau), the speech
model learning the structure of the sound it hears (loss falling), two made-up languages being
discovered as separate clusters, voice-to-voice replies in each, and a self-training round.
Measured on 4 CPU cores with the `tiny` preset and 60 s of audio: log-mel distance 3.96 to
2.66 after 400 codec steps, speech-model loss 4.2 to 1.05 after 200 steps.

### Training on real recordings

Any wav files work (8/16/24/32-bit, mono or stereo, any rate; everything is resampled to
16 kHz mono). Three presets: `tiny` (tests/demo), `cpu` (a few CPU cores, minutes of audio,
under an hour), `small` (hours of audio, a GPU helps).

No recordings at hand? Two sources that need nothing but a network connection:

```bash
# 3,000 real recordings of spoken digits by six speakers (Free Spoken Digit Dataset, 8 kHz)
mkdir -p data/fsdd && cd data/fsdd
for d in 0 1 2 3 4 5 6 7 8 9; do for s in jackson nicolas theo yweweler george lucas; do for i in $(seq 0 49); do
  curl -sfO "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/${d}_${s}_${i}.wav"
done; done; done; cd ../..

# a multilingual supplement from the offline espeak-ng synthesizer (synthetic, but real phoneme sequences)
pip install espeakng-loader
python3 scripts/espeak_corpus.py data/espeak --languages en hi es de fr ja ar sw --per-language 30
```

Then:

```bash
voicecore init ~/vc --preset cpu
voicecore train-codec ~/vc data/fsdd data/espeak --steps 3000     # ~20 min on 4 cores
voicecore train-lm    ~/vc data/fsdd data/espeak --steps 2000     # ~3 min
voicecore selftrain   ~/vc --rounds 2 --audio data/fsdd
voicecore roundtrip   ~/vc data/fsdd/7_jackson_0.wav seven_roundtrip.wav
voicecore talk        ~/vc data/fsdd/7_jackson_0.wav reply.wav
```

With your own recordings (any language, 16 kHz mono wav is best; anything is resampled):

```bash
voicecore init ~/vc --preset small                        # or --preset cpu without a GPU
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

## Web UI

```bash
voicecore serve ~/vc --open        # http://localhost:8765
```

Record from the microphone or upload a clip, hear the voice reply or the codec round trip, see
which voice cluster it was assigned to, rate replies, run a self-training round with live
progress, and refit or name discovered voices. The page is served by the standard library; no
extra packages. The same endpoints are a plain HTTP API (`voicecore/server.py`).

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
