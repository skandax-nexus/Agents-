# VoiceCore architecture

## Why voice-to-voice and not STT -> LLM -> TTS

A text pipeline throws away everything that is not words: tone, emotion, accent, hesitation,
who is speaking, and it needs a transcription model and a synthesis model *per language*.
Modelling audio tokens directly keeps all of that and is language-agnostic: there is no
vocabulary of words, only a vocabulary of sounds learned from the audio itself.

## Components

### 1. Audio codec (`voicecore/codec`) - learning to hear and to speak sounds
- Encoder: strided 1-D convolutions with residual blocks, 16 kHz waveform -> one latent
  vector per 20 ms (strides 2·4·5·8 = 320 samples).
- Residual vector quantisation: `n_codebooks` codebooks (default 4 × 1024). Each frame
  becomes 4 integers. EMA codebook updates, dead-code re-seeding, commitment loss.
- Decoder: mirror of the encoder with transposed convolutions.
- Loss: waveform MSE (dominant; frames are time-aligned so this is a strong signal from step
  one) plus a small multi-resolution spectral term (scale-free linear-magnitude L1 and clamped
  log-mel L1). Measured on this architecture, a large log-spectral weight collapses early
  training toward silence and codebook collapse, so the spectral weight is 0.1 and its
  gradients are bounded. Self-supervised: the target is the input. (A GAN discriminator, as
  in EnCodec, would sharpen output and is the natural next step.)
- Anti-collapse: data-dependent codebook init decided once at the first training step (never
  re-run, so a trained codebook is never wiped on resume), l2-normalised (cosine) codes, and
  re-seeding of any code unused for 10 steps. Usage sits at a handful of codes for the first
  ~150 steps and then diversifies. Quality metrics: SNR, log-mel distance, and codes in use.
- Inference helpers (`encode`, `decode`, `embed`, `fingerprint`) run in eval mode under
  `no_grad`, so they never touch the EMA codebooks even right after a training step.

### 2. Speech language model (`voicecore/lm`) - learning the structure of speech
- Decoder-only transformer over codec frames. Each frame's codes are embedded (one table
  per codebook, summed) and the model predicts the next frame's codes with one head per
  codebook (parallel pattern). A delay-pattern or depth transformer would raise quality.
- A **style prefix**: the codec's mean latent of the heard utterance, projected and prepended
  as frame 0. It carries voice, prosody and language, so a reply comes out in the language
  and voice it was spoken to in. Nothing is told which language it is.
- Two training signals, no text anywhere:
  - *continuation*: `[BOS] audio...` next-frame prediction on everything heard;
  - *exchange*: `[BOS] heard [SEP] reply [EOS]` with the loss on the reply (from
    conversation recordings split on silence, from live exchanges, from self-play).

### 3. Engine (`voicecore/engine.py`)
`hear` stores and clusters; `respond` encodes, generates `[BOS] heard [SEP] -> reply`, decodes,
streams chunks through `on_chunk`, and records the exchange. `round_trip` is codec-only.

### 4. Voice/language discovery (`voicecore/lang`)
Online k-means over an acoustic fingerprint (log-mel mean and spread: pitch, timbre, sound
inventory). Pure signal processing, so it works from the first utterance; label clusters
with `voicecore name`. It is discovery, not a trained language-ID model.

### 5. Self-training (`voicecore/selftrain`)
Every round: codec update on all heard audio -> curriculum gate (codec SNR must pass a bar)
-> LM continuation update -> self-play (generate several replies to recent utterances, score
with the model's own reply likelihood + codec round-trip consistency + non-silence, keep the
best) -> dialogue update on live exchanges, kept self-play exchanges and approved replies
(positive feedback ×3, negative excluded) -> cluster refresh -> save.

This is self-supervised + self-play learning with the model as its own critic, the same
family of ideas as STaR / self-play RL, applied to sound. It cannot invent language it has
never heard; it gets better at the sound and structure of what it hears.

## Scaling guide

| stage | data | compute | what you get |
|---|---|---|---|
| codec `tiny` | minutes | CPU minutes | demo-quality reconstruction of synthetic speech |
| codec `small` | 5-50 h any speech | 1 GPU, hours | intelligible reconstruction of real voices |
| LM `small` continuation | 50-500 h | 1 GPU, days | voice-like babble with the right prosody and language |
| LM exchange + self-play | 100s of h of conversation | GPUs, weeks | short, situationally plausible replies |
| fluent dialogue | 10k h+ | large cluster | what commercial speech-to-speech systems train on |

Nothing in the code changes between rows: only `VoiceCoreConfig` sizes, data and steps.
