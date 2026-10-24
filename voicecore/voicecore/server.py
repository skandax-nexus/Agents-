"""Local web UI and HTTP API for a VoiceCore engine (standard library only).

    voicecore serve ~/vc --port 8765      ->  http://localhost:8765

API (all audio is WAV; the browser page records from the microphone and converts to 16-bit WAV):
    GET  /healthz                        liveness: {"ok": true, ...}
    GET  /api/status                     engine status, clusters, self-training state
    POST /api/respond?learn=1            body: wav  -> reply wav; headers X-Cluster, X-Cluster-Name, X-Exchange-Id
    POST /api/roundtrip                  body: wav  -> codec round-trip wav
    POST /api/hear                       body: wav  -> {cluster, ...}
    POST /api/feedback?id=..&value=1|-1  rate a reply
    POST /api/selftrain                  start one self-training round in the background
    POST /api/refit?k=N                  re-cluster stored fingerprints
    POST /api/name?cluster=N&label=..    name a cluster

Security model: binds to 127.0.0.1 by default.  To expose it beyond the machine, pass --host and a
--token; every /api/* request must then carry `Authorization: Bearer <token>`.  Uploads are capped
(--max-upload-mb) and audio longer than the engine's context is truncated to its tail.
"""
from __future__ import annotations

import hmac
import io
import json
import logging
import signal
import socket
import struct
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import numpy as np

from . import __version__
from .audio.io import to_mono
from .engine import VoiceCore
from .lang.cluster import clean_label

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("voicecore.server")


class RequestError(ValueError):
    """A client error with an HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def wav_bytes_to_audio(data: bytes, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    """Decode a PCM WAV upload.  With max_seconds, only the last max_seconds of frames are read, so the
    decoded arrays stay small however large the upload is (the raw body is the only full-size copy)."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            sr, ch, width, n = wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()
            if sr <= 0 or ch <= 0 or width not in (1, 2, 3, 4):
                raise RequestError(400, "unsupported WAV format (need PCM 8/16/24/32-bit)")
            if max_seconds is not None:
                keep = int(max_seconds * sr)
                if n > keep:
                    wf.setpos(n - keep)
                    n = keep
            raw = wf.readframes(n)
    except (wave.Error, EOFError, struct.error) as exc:
        raise RequestError(400, f"not a PCM WAV file: {exc}") from exc
    frame_bytes = width * ch
    raw = raw[: len(raw) - len(raw) % frame_bytes]      # a truncated final frame is dropped, not a crash
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        x = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2].astype(np.int8).astype(np.int32) << 16)).astype(np.float32) / 8388608.0
    else:
        raise RequestError(400, f"unsupported sample width {width}")
    if ch > 1:
        x = to_mono(x.reshape(-1, ch))
    if len(x) == 0:
        raise RequestError(400, "WAV file contains no samples")
    return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), sr


def audio_to_wav_bytes(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(x, -1, 1) * 32767.0).astype("<i2").tobytes())
    return buf.getvalue()


class EngineService:
    """Serialises access to the engine (torch models are not safe for concurrent use) and owns the
    background self-training job."""

    def __init__(self, engine: VoiceCore, *, token: str | None = None, max_upload_bytes: int = 32 << 20,
                 max_input_seconds: float = 30.0) -> None:
        self.engine = engine
        if token is not None and (not token.isascii() or not token.isprintable() or len(token) < 8):
            raise ValueError("token must be at least 8 printable ASCII characters")
        self.token = token.encode("ascii") if token else None
        self.max_upload_bytes = max_upload_bytes
        self.max_input_seconds = max_input_seconds
        self.lock = threading.Lock()
        self.selftrain = {"running": False, "rounds": 0, "last": None, "error": None, "started": None}
        self.events: list[dict] = []
        self.started = time.time()
        self.requests = 0
        self.errors = 0
        self.stop_event = threading.Event()
        self.train_thread: threading.Thread | None = None

    def _clip(self, wav: np.ndarray, sr: int) -> np.ndarray:
        limit = int(self.max_input_seconds * sr)
        return wav[-limit:] if len(wav) > limit else wav

    def health(self) -> dict:
        return {"ok": True, "version": __version__, "engine": str(self.engine.root), "uptime_seconds": round(time.time() - self.started),
                "requests": self.requests, "errors": self.errors, "selftrain_running": self.selftrain["running"]}

    def status(self) -> dict:
        with self.lock:
            st = self.engine.status()
        st["selftrain"] = self.selftrain
        st["fingerprint"] = "trained codec latent" if self.engine.codec_trained else "acoustic"
        st["fingerprints_stored"] = len(self.engine.clusters.fingerprints)
        st["version"] = __version__
        return st

    def respond(self, wav: np.ndarray, sr: int, learn: bool) -> dict:
        with self.lock:
            return self.engine.converse(self._clip(wav, sr), sr, learn=learn)

    def roundtrip(self, wav: np.ndarray, sr: int) -> np.ndarray:
        with self.lock:
            return self.engine.round_trip(self._clip(wav, sr), sr)

    def hear(self, wav: np.ndarray, sr: int) -> dict:
        with self.lock:
            return self.engine.hear(self._clip(wav, sr), sr, source="web")

    def feedback(self, exchange_id: str, value: int) -> bool:
        with self.lock:
            return self.engine.feedback(exchange_id, value)

    def refit(self, k: int | None) -> dict:
        if k is not None and not (1 <= k <= 64):
            raise RequestError(400, "k must be between 1 and 64")
        with self.lock:
            info = self.engine.clusters.refit(k)
            self.engine.save_light()
            return info

    def name(self, cluster: int, label: str) -> list[dict]:
        try:
            label = clean_label(label)
        except ValueError as exc:
            raise RequestError(400, str(exc)) from exc
        with self.lock:
            try:
                self.engine.clusters.name(cluster, label)
            except IndexError as exc:
                raise RequestError(404, str(exc)) from exc
            self.engine.save_light()
            return self.engine.clusters.summary()

    def start_selftrain(self) -> bool:
        with self.lock:
            if self.selftrain["running"]:
                return False
            self.selftrain.update({"running": True, "error": None, "started": time.time()})
            self.events = []

        def run() -> None:
            from .selftrain.loop import SelfTrainer

            try:
                with self.lock:
                    trainer = SelfTrainer(self.engine, stop_event=self.stop_event,
                                          on_event=lambda k, d: self.events.append({"kind": k, **d})
                                          if k in ("codec_step", "lm_step", "selftrain_round", "selftrain_skipped") else None)
                    report = trainer.round()
                self.selftrain["last"] = report
                self.selftrain["rounds"] += 1
                log.info("self-training round finished: %s", {k: report.get(k) for k in ("stage", "codec_snr_db", "lm_loss", "note")})
            except Exception as exc:  # noqa: BLE001 - surfaced to the client, server keeps running
                self.selftrain["error"] = f"{type(exc).__name__}: {exc}"
                log.exception("self-training round failed")
            finally:
                self.selftrain["running"] = False

        self.train_thread = threading.Thread(target=run, name="voicecore-selftrain", daemon=True)
        self.train_thread.start()
        return True

    def shutdown(self, timeout: float = 60.0) -> None:
        """Ask a running self-training round to stop between steps, wait for it, then save."""
        self.stop_event.set()
        t = self.train_thread
        if t is not None and t.is_alive():
            log.info("waiting up to %.0fs for the self-training round to stop", timeout)
            t.join(timeout)
            if t.is_alive():
                log.warning("self-training did not stop in time; its own round-end save is the last state")
                return
        with self.lock:
            self.engine.save()

    def save(self) -> None:
        with self.lock:
            self.engine.save()


def make_handler(service: EngineService):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"VoiceCore/{__version__}"
        protocol_version = "HTTP/1.1"
        timeout = 30          # socket timeout: a stalled or idle connection releases its thread

        def log_message(self, fmt, *args):
            log.debug("%s %s", self.address_string(), fmt % args)

        # -- helpers ------------------------------------------------------------------
        def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def _authorised(self) -> bool:
            if not service.token:
                return True
            header = self.headers.get("Authorization", "")
            supplied = header[7:] if header.startswith("Bearer ") else ""
            return hmac.compare_digest(supplied.encode("latin-1", "replace"), service.token)

        def _same_origin(self) -> bool:
            """Reject state-changing requests that a browser marks as coming from another site, so a web
            page elsewhere cannot drive a token-less localhost server through the visitor's browser."""
            site = self.headers.get("Sec-Fetch-Site", "")
            if site in ("cross-site", "same-site"):
                return False
            origin = self.headers.get("Origin")
            if origin and origin != "null":
                host = self.headers.get("Host", "")
                return urlparse(origin).netloc.lower() == host.lower()
            return True

        def _read_body(self) -> bytes:
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise RequestError(400, "bad Content-Length") from exc
            if n <= 0:
                raise RequestError(400, "empty body: send a WAV file")
            if n > service.max_upload_bytes:
                raise RequestError(413, f"upload larger than {service.max_upload_bytes >> 20} MB")
            data = self.rfile.read(n)
            if len(data) != n:
                raise RequestError(400, "truncated upload")
            return data

        def _read_audio(self) -> tuple[np.ndarray, int]:
            return wav_bytes_to_audio(self._read_body(), max_seconds=service.max_input_seconds)

        def _drain(self) -> None:
            """Discard an unread body so the keep-alive connection stays in sync."""
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = 0
            while n > 0:
                chunk = self.rfile.read(min(n, 1 << 16))
                if not chunk:
                    break
                n -= len(chunk)

        # -- routes -------------------------------------------------------------------
        def do_HEAD(self) -> None:
            self.do_GET()

        def do_GET(self) -> None:
            url = urlparse(self.path)
            service.requests += 1
            self._drain()            # a GET carrying a body must not desync the keep-alive connection
            if url.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/healthz":
                self._json(service.health())
            elif url.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            elif url.path.startswith("/api/"):
                if not self._authorised():
                    self._json({"error": "unauthorised"}, 401)
                elif url.path == "/api/status":
                    self._json(service.status())
                elif url.path == "/api/selftrain":
                    self._json({**service.selftrain, "events": service.events[-20:]})
                else:
                    self._json({"error": "not found"}, 404)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            service.requests += 1
            if not url.path.startswith("/api/"):
                self._drain()
                self._json({"error": "not found"}, 404)
                return
            if not self._authorised():
                self._drain()
                self._json({"error": "unauthorised"}, 401)
                return
            if not self._same_origin():
                self._drain()
                self._json({"error": "cross-site request refused"}, 403)
                return
            started = time.time()
            try:
                if url.path == "/api/respond":
                    wav, sr = self._read_audio()
                    info = service.respond(wav, sr, learn=q.get("learn", "1") not in ("0", "false"))
                    self._send(200, audio_to_wav_bytes(info["reply"], service.engine.sr), "audio/wav", {
                        "X-Cluster": info["cluster"] if info["cluster"] is not None else "",
                        # percent-encoded so any name (any script, any character) is a safe ASCII header value
                        "X-Cluster-Name": quote(info["cluster_name"], safe=""), "X-Exchange-Id": info["exchange_id"] or "",
                        "X-Reply-Seconds": f"{info['reply_seconds']:.2f}",
                        "Access-Control-Expose-Headers": "X-Cluster, X-Cluster-Name, X-Exchange-Id, X-Reply-Seconds"})
                elif url.path == "/api/roundtrip":
                    wav, sr = self._read_audio()
                    self._send(200, audio_to_wav_bytes(service.roundtrip(wav, sr), service.engine.sr), "audio/wav")
                elif url.path == "/api/hear":
                    wav, sr = self._read_audio()
                    self._json(service.hear(wav, sr))
                elif url.path == "/api/feedback":
                    self._drain()
                    value = q.get("value", "1")
                    if value not in ("1", "+1", "up", "-1", "down"):
                        raise RequestError(400, "value must be +1 or -1")
                    ok = service.feedback(q.get("id", ""), 1 if value in ("1", "+1", "up") else -1)
                    self._json({"ok": ok}, 200 if ok else 404)
                elif url.path == "/api/selftrain":
                    self._drain()
                    self._json({"started": service.start_selftrain(), **service.selftrain})
                elif url.path == "/api/refit":
                    self._drain()
                    try:
                        k = int(q["k"]) if q.get("k") else None
                    except ValueError as exc:
                        raise RequestError(400, "k must be an integer") from exc
                    self._json(service.refit(k))
                elif url.path == "/api/name":
                    self._drain()
                    try:
                        cluster = int(q.get("cluster", ""))
                    except ValueError as exc:
                        raise RequestError(400, "cluster must be an integer") from exc
                    self._json({"ok": True, "clusters": service.name(cluster, q.get("label", ""))})
                else:
                    self._drain()
                    self._json({"error": "not found"}, 404)
                log.info("%s %s %.0f ms", self.command, url.path, (time.time() - started) * 1000)
            except RequestError as exc:
                service.errors += 1
                self._json({"error": str(exc)}, exc.status)
            except (BrokenPipeError, ConnectionResetError):
                log.info("client went away during %s", url.path)
            except Exception:  # noqa: BLE001 - keep serving; details stay in the server log
                service.errors += 1
                log.exception("unhandled error in %s", url.path)
                self._json({"error": "internal error (see server log)"}, 500)

    return Handler


LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler):
        self.address_family = socket.getaddrinfo(address[0], address[1], type=socket.SOCK_STREAM)[0][0]
        super().__init__(address, handler)


def serve(root: str | Path, host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = False,
          token: str | None = None, max_upload_mb: int = 32, log_level: str = "info",
          init_preset: str | None = None) -> None:
    """Run the web UI/API.  `token` falls back to the VOICECORE_TOKEN environment variable; `init_preset`
    creates the engine with that preset when the directory holds none (first start in a container)."""
    import os

    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    token = token or os.environ.get("VOICECORE_TOKEN") or None
    if host not in LOCAL_HOSTS and not token:
        raise SystemExit("refusing to listen on a non-local address without --token (or VOICECORE_TOKEN)")
    root = Path(root).expanduser()
    if init_preset and not (root / "config.json").exists():
        from .config import VoiceCoreConfig

        presets = {"tiny": VoiceCoreConfig.tiny, "cpu": VoiceCoreConfig.cpu, "small": VoiceCoreConfig.small,
                   "default": VoiceCoreConfig}
        if init_preset not in presets:
            raise SystemExit(f"unknown preset {init_preset!r}; choose one of {', '.join(presets)}")
        log.info("no engine at %s: creating one with the %s preset (random weights)", root, init_preset)
        VoiceCore(root, presets[init_preset]()).save()
    engine = VoiceCore(root)
    try:
        service = EngineService(engine, token=token, max_upload_bytes=max_upload_mb << 20)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    httpd = _Server((host, port), make_handler(service))
    url = f"http://{host}:{port}" if ":" not in host else f"http://[{host}]:{port}"
    log.info("VoiceCore %s serving %s (engine %s, %s)", __version__, url, engine.root,
             "codec trained" if engine.codec_trained else "untrained")
    print(f"VoiceCore web UI at {url}  (Ctrl-C to stop)", flush=True)
    if open_browser:
        import webbrowser

        webbrowser.open(url)

    stop = threading.Event()

    def on_signal(signum, frame):  # noqa: ARG001
        log.info("signal %s: shutting down", signum)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, on_signal)
        except (ValueError, OSError):
            pass  # not the main thread (tests)
    server_thread = threading.Thread(target=httpd.serve_forever, name="voicecore-http", daemon=True)
    server_thread.start()
    try:
        while not stop.is_set():
            stop.wait(0.5)
    finally:
        httpd.shutdown()
        httpd.server_close()
        service.shutdown()
        log.info("engine saved, server stopped")
