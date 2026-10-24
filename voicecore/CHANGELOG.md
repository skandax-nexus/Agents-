# Changelog

## 0.1.0

First release of the core engine.

- Neural audio codec (conv encoder, residual VQ with EMA codebooks, decoder) trained from scratch
- Speech language model over codec tokens with a style prefix; continuation and exchange training;
  streamed generation
- Voice-to-voice engine: hear, respond, converse, stream, round trip, feedback
- Unsupervised voice/language discovery with stored fingerprints and batch refit
- Self-training rounds: codec + LM updates, curriculum gate, self-play with self-scoring,
  feedback-weighted dialogue updates
- CLI: init, train-codec, train-lm, train-dialogue, talk, roundtrip, listen, selftrain, langs,
  name, feedback, status, serve, demo
- Local web UI and HTTP API with bearer-token auth, upload limits, health endpoint, graceful shutdown
- Presets: tiny (tests), cpu (laptop), small (GPU)
