"""Local web UI and HTTP API for a VoiceCore engine (standard library only).

    voicecore serve ~/vc --port 8765      ->  http://localhost:8765

API (all audio is WAV; the browser page records from the microphone and converts to 16-bit WAV):
    GET  /api/status                     engine status, clusters, self-training state
    POST /api/respond?learn=1            body: wav  -> reply wav; headers X-Cluster, X-Cluster-Name, X-Exchange-Id
    POST /api/roundtrip                  body: wav  -> codec round-trip wav
    POST /api/hear                       body: wav  -> {cluster, ...}
    POST /api/feedback?id=..&value=1|-1  rate a reply
    POST /api/selftrain                  start one self-training round in the background
    POST /api/refit?k=N                  re-cluster stored fingerprints
    POST /api/name?cluster=N&label=..    name a cluster
"""
from __future__ import annotations

import io
import json
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .audio.io import resample, to_mono, write_wav
from .engine import VoiceCore

STATIC = Path(__file__).parent / "static"


def wav_bytes_to_audio(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr, ch, width, n = wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()
        raw = wf.readframes(n)
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
        raise ValueError(f"unsupported sample width {width}")
    if ch > 1:
        x = to_mono(x.reshape(-1, ch))
    return x, sr


def audio_to_wav_bytes(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(x, -1, 1) * 32767.0).astype("<i2").tobytes())
    return buf.getvalue()


class EngineService:
    """Serialises access to the engine (torch models are not safe for concurrent use)."""

    def __init__(self, engine: VoiceCore) -> None:
        self.engine = engine
        self.lock = threading.Lock()
        self.selftrain = {"running": False, "rounds": 0, "last": None, "error": None}
        self.events: list[dict] = []

    def status(self) -> dict:
        with self.lock:
            st = self.engine.status()
        st["selftrain"] = self.selftrain
        st["fingerprint"] = "trained codec latent" if self.engine.codec_trained else "acoustic"
        st["fingerprints_stored"] = len(self.engine.clusters.fingerprints)
        return st

    def respond(self, wav: np.ndarray, sr: int, learn: bool) -> dict:
        with self.lock:
            return self.engine.converse(wav, sr, learn=learn)

    def roundtrip(self, wav: np.ndarray, sr: int) -> np.ndarray:
        with self.lock:
            return self.engine.round_trip(wav, sr)

    def hear(self, wav: np.ndarray, sr: int) -> dict:
        with self.lock:
            return self.engine.hear(wav, sr, source="web")

    def feedback(self, exchange_id: str, value: int) -> bool:
        with self.lock:
            return self.engine.feedback(exchange_id, value)

    def refit(self, k: int | None) -> dict:
        with self.lock:
            info = self.engine.clusters.refit(k)
            self.engine.save_light()
            return info

    def name(self, cluster: int, label: str) -> None:
        with self.lock:
            self.engine.clusters.name(cluster, label)
            self.engine.save_light()

    def start_selftrain(self) -> bool:
        if self.selftrain["running"]:
            return False
        self.selftrain.update({"running": True, "error": None, "started": time.time()})
        self.events = []

        def run() -> None:
            from .selftrain.loop import SelfTrainer

            try:
                with self.lock:
                    trainer = SelfTrainer(self.engine, on_event=lambda k, d: self.events.append({"kind": k, **d}) if k in ("codec_step", "lm_step", "selftrain_round", "selftrain_skipped") else None)
                    report = trainer.round()
                self.selftrain["last"] = report
                self.selftrain["rounds"] += 1
            except Exception as exc:  # noqa: BLE001
                self.selftrain["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                self.selftrain["running"] = False

        threading.Thread(target=run, daemon=True).start()
        return True


def make_handler(service: EngineService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "VoiceCore/0.1"

        def log_message(self, fmt, *args):  # quieter log
            if "/api/" in (args[0] if args else ""):
                print(f"  {self.address_string()} {fmt % args}")

        # -- helpers ------------------------------------------------------------------
        def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def _read_audio(self) -> tuple[np.ndarray, int]:
            n = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(n)
            return wav_bytes_to_audio(data)

        # -- routes -------------------------------------------------------------------
        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/status":
                self._json(service.status())
            elif url.path == "/api/selftrain":
                self._json({**service.selftrain, "events": service.events[-20:]})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/api/respond":
                    wav, sr = self._read_audio()
                    info = service.respond(wav, sr, learn=q.get("learn", "1") not in ("0", "false"))
                    self._send(200, audio_to_wav_bytes(info["reply"], service.engine.sr), "audio/wav", {
                        "X-Cluster": info["cluster"] if info["cluster"] is not None else "",
                        "X-Cluster-Name": info["cluster_name"], "X-Exchange-Id": info["exchange_id"] or "",
                        "X-Reply-Seconds": f"{info['reply_seconds']:.2f}",
                        "Access-Control-Expose-Headers": "X-Cluster, X-Cluster-Name, X-Exchange-Id, X-Reply-Seconds"})
                elif url.path == "/api/roundtrip":
                    wav, sr = self._read_audio()
                    self._send(200, audio_to_wav_bytes(service.roundtrip(wav, sr), service.engine.sr), "audio/wav")
                elif url.path == "/api/hear":
                    wav, sr = self._read_audio()
                    self._json(service.hear(wav, sr))
                elif url.path == "/api/feedback":
                    ok = service.feedback(q.get("id", ""), 1 if q.get("value", "1") in ("1", "+1", "up") else -1)
                    self._json({"ok": ok})
                elif url.path == "/api/selftrain":
                    self._json({"started": service.start_selftrain(), **service.selftrain})
                elif url.path == "/api/refit":
                    self._json(service.refit(int(q["k"]) if q.get("k") else None))
                elif url.path == "/api/name":
                    service.name(int(q["cluster"]), q.get("label", "")[:40])
                    self._json({"ok": True, "clusters": service.engine.clusters.summary()})
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as exc:  # noqa: BLE001 - report to the client, keep serving
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

    return Handler


def serve(root: str | Path, host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = False) -> None:
    engine = VoiceCore(root)
    service = EngineService(engine)
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    url = f"http://{host}:{port}"
    print(f"VoiceCore web UI at {url}  (engine {engine.root}, {'codec trained' if engine.codec_trained else 'untrained'}; Ctrl-C to stop)")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with service.lock:
            engine.save()
        httpd.server_close()
        print("saved and stopped")
