# Running VoiceCore in production

What "production" means here, stated plainly: the **software** is engineered to be run and
operated reliably (tested, linted, containerised, health-checked, authenticated, with atomic
saves and graceful shutdown). The **model quality** is whatever your data and compute bought.
A 20-minute CPU run on 17 minutes of digits gives intelligible codec round trips and
voice-like replies; fluent speech needs hours of audio and GPU-days (see `ARCHITECTURE.md`).

## Deployment

```bash
docker build -t voicecore .
docker run -d --name voicecore --restart unless-stopped \
  -p 127.0.0.1:8765:8765 -v voicecore-engine:/data \
  -e VOICECORE_TOKEN="$(openssl rand -hex 24)" voicecore --init-preset cpu
curl -s http://127.0.0.1:8765/healthz
```

`--init-preset` creates the engine on the first start only (an unknown preset is a hard error,
nothing is created). The process is PID 1 (exec-form entrypoint), so `docker stop` delivers
SIGTERM straight to it. Open the page with `?token=<token>` or paste the token into the Engine
panel when prompted; it is kept for that browser tab only.

- The engine (weights, experience buffer, clusters) lives in the `/data` volume. Back it up by
  snapshotting the volume; everything is plain files and restores by copying them back.
- Put a reverse proxy (nginx, Caddy) with TLS in front for anything beyond localhost. The
  server refuses to bind a non-local address without `--token`/`VOICECORE_TOKEN`, and every
  `/api/*` call must carry `Authorization: Bearer <token>` (constant-time compared; tokens are
  8+ printable ASCII characters). The HTML page itself is public on that port; keep the proxy in
  front of it too. State-changing requests that a browser marks as cross-site (`Sec-Fetch-Site`,
  mismatched `Origin`) are refused with 403, so a web page elsewhere cannot drive a token-less
  localhost server through a visitor's browser.
- Cluster names are limited to 40 printable characters (no control characters) and are
  percent-encoded in response headers and rendered as text, never as HTML.
- One process serves one engine: `voicecore serve` takes an exclusive lock on the engine
  directory and a second server on the same directory refuses to start. Requests are serialised
  through a lock (torch models are not safe for concurrent use), so throughput is one reply at a time; replies take 0.5 to 2 s on CPU
  for the `cpu` preset. Run several containers behind the proxy for more concurrency; they do
  not share learning.
- Self-training runs in a background thread inside the server and holds the engine lock for the
  whole round, during which replies wait. Schedule rounds off-peak, or train in a separate
  process with `voicecore selftrain` and restart the server to pick up the new weights.

## Limits and validation

| Guard | Default | Flag |
|---|---|---|
| Upload size | 32 MB | `--max-upload-mb` |
| Input length | last 30 s decoded; the rest is never expanded in memory | `EngineService(max_input_seconds=...)` |
| Connection | 30 s socket timeout per read | `Handler.timeout` |
| Reply length | `max_response_seconds` (8 s) | config |
| Bind address | 127.0.0.1 | `--host` (+ `--token`) |

Malformed or truncated WAV, bad parameters and unknown clusters return 4xx JSON with a message;
unexpected errors return a generic 500 (details only in the server log, with a traceback); the
server keeps serving. GET requests carrying a body are drained so keep-alive connections never
desynchronise.

## Operations

- `GET /healthz`: liveness with uptime, request and error counters, whether self-training runs.
- Logs go to stderr (`--log-level debug|info|warning|error`), one line per API call with latency.
- `SIGTERM`/`SIGINT` stop accepting connections, ask a running self-training round to stop at
  its next step (waiting up to 60 s), save the engine atomically and exit 0. Every state file
  (checkpoints, config, clusters, counters, the experience index) is written to a temporary
  file, fsynced and renamed, so neither a crash nor a power loss leaves a torn file; the
  experience index tolerates a half-written last line and unreadable clips are skipped with a
  warning rather than breaking status or training.
- Resource needs for the `cpu` preset: ~1 GB RAM, 2+ cores; the `small` preset wants a GPU.

## Upgrades

Engine directories carry `config.json`; the engine refuses to load weights that do not match
the config, and `voicecore init` refuses to overwrite an existing engine without `--force`.
Checkpoints record a `version` field for future migrations. Pin `voicecore` to a release in
your deployment and read `CHANGELOG.md` before upgrading.

## Privacy

Everything the engine hears is stored under `experience/` in the engine directory and used for
learning. Say so to the people who talk to it, give them `--no-learn`/the "learn" checkbox, and
delete `experience/` to forget.
