"""The local web API: status, respond, round trip, feedback, refit, naming."""
import io
import json
import threading
import urllib.parse
import urllib.request
import wave
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from voicecore.config import VoiceCoreConfig
from voicecore.data import proto_speech
from voicecore.engine import VoiceCore
from voicecore.server import (
    EngineService,
    audio_to_wav_bytes,
    make_handler,
    wav_bytes_to_audio,
)


@pytest.fixture
def server(tmp_path):
    cfg = VoiceCoreConfig.tiny(); cfg.max_response_seconds = 0.5
    engine = VoiceCore(tmp_path / "vc", cfg)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(EngineService(engine)))
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", engine
    httpd.shutdown()


def _post(url, data=None, ctype="audio/wav"):
    req = urllib.request.Request(url, data=data if data is not None else b"", method="POST", headers={"Content-Type": ctype})
    return urllib.request.urlopen(req, timeout=60)


def test_wav_helpers_roundtrip():
    x = proto_speech(0.3, seed=1)
    y, sr = wav_bytes_to_audio(audio_to_wav_bytes(x, 16000))
    assert sr == 16000 and np.abs(y - x).max() < 1e-3


def test_api_flow(server):
    base, engine = server
    page = urllib.request.urlopen(base + "/", timeout=10).read().decode()
    assert "<title>VoiceCore</title>" in page and "/api/respond" in page
    st = json.loads(urllib.request.urlopen(base + "/api/status", timeout=10).read())
    assert st["sample_rate"] == 16000 and st["fingerprint"] == "acoustic"
    wav = audio_to_wav_bytes(proto_speech(0.8, seed=2), 16000)
    r = _post(base + "/api/respond?learn=1", wav)
    reply = r.read()
    assert r.headers["Content-Type"] == "audio/wav" and r.headers["X-Cluster-Name"].startswith("cluster-")
    y, sr = wav_bytes_to_audio(reply)
    assert sr == 16000 and 0 < len(y) <= 8000 + 320
    exchange = r.headers["X-Exchange-Id"]
    assert exchange and json.loads(_post(base + f"/api/feedback?id={exchange}&value=1").read())["ok"]
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(base + "/api/feedback?id=nope&value=1")
    assert exc.value.code == 404 and json.loads(exc.value.read())["ok"] is False
    # a 44.1 kHz stereo upload is accepted
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(44100)
        x = (np.tile(proto_speech(0.5, sr=44100, seed=3)[:, None], (1, 2)) * 32767).astype("<i2")
        wf.writeframes(x.tobytes())
    y2, _ = wav_bytes_to_audio(_post(base + "/api/roundtrip", buf.getvalue()).read())
    assert abs(len(y2) - 8000) <= 320
    _post(base + "/api/hear", wav)
    info = json.loads(_post(base + "/api/refit?k=2").read())
    assert info["clusters"] == 2
    named = json.loads(_post(base + "/api/name?cluster=0&label=Alice").read())
    assert named["clusters"][0]["name"] == "Alice"
    # malformed audio is a 400 with a message, and the server keeps serving
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(base + "/api/respond", b"not a wav")
    assert exc.value.code == 400 and "error" in json.loads(exc.value.read())
    assert json.loads(urllib.request.urlopen(base + "/api/status", timeout=10).read())["experience"]["exchanges"] == 1


def test_health_and_limits(tmp_path):
    cfg = VoiceCoreConfig.tiny(); cfg.max_response_seconds = 0.3
    engine = VoiceCore(tmp_path / "vc", cfg)
    service = EngineService(engine, token="s3cret-token", max_upload_bytes=20_000, max_input_seconds=1.0)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        h = json.loads(urllib.request.urlopen(base + "/healthz", timeout=10).read())
        assert h["ok"] and h["version"] and "uptime_seconds" in h
        # the page is served without a token, the API is not
        assert urllib.request.urlopen(base + "/", timeout=10).status == 200
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/api/status", timeout=10)
        assert exc.value.code == 401
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(base + "/api/respond", audio_to_wav_bytes(proto_speech(0.3, seed=1), 16000))
        assert exc.value.code == 401
        auth = {"Authorization": "Bearer s3cret-token"}
        req = urllib.request.Request(base + "/api/status", headers=auth)
        assert json.loads(urllib.request.urlopen(req, timeout=10).read())["version"]
        # oversized upload -> 413, then the connection is still usable
        big = audio_to_wav_bytes(proto_speech(2.0, seed=1), 16000)
        req = urllib.request.Request(base + "/api/respond", data=big, method="POST", headers={**auth, "Content-Type": "audio/wav"})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 413
        # a long input is truncated to its tail (max_input_seconds) and still answered
        wav = audio_to_wav_bytes(proto_speech(0.5, seed=2), 16000)
        req = urllib.request.Request(base + "/api/roundtrip", data=wav, method="POST", headers={**auth, "Content-Type": "audio/wav"})
        y, _ = wav_bytes_to_audio(urllib.request.urlopen(req, timeout=60).read())
        assert abs(len(y) - 8000) <= 320
        # bad parameters are 4xx with a message
        for path, code in (("/api/feedback?id=x&value=7", 400), ("/api/refit?k=999", 400), ("/api/name?cluster=9&label=x", 404),
                           ("/api/name?cluster=abc&label=x", 400), ("/api/feedback?id=missing&value=1", 404)):
            req = urllib.request.Request(base + path, data=b"", method="POST", headers=auth)
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(req, timeout=10)
            assert exc.value.code == code, path
            assert "error" in json.loads(exc.value.read()) or code == 404
        assert json.loads(urllib.request.urlopen(base + "/healthz", timeout=10).read())["errors"] >= 5
    finally:
        httpd.shutdown()


def test_atomic_saves_leave_no_temp_files(tmp_path):
    cfg = VoiceCoreConfig.tiny()
    engine = VoiceCore(tmp_path / "vc", cfg)
    engine.hear(proto_speech(0.5, seed=1))
    engine.save()
    names = {p.name for p in (tmp_path / "vc").iterdir()}
    assert {"codec.pt", "lm.pt", "clusters.json", "clusters.npy", "meta.json", "config.json"} <= names
    assert not any(n.endswith(".tmp") for n in names)
    # a half-written last line in the experience index is skipped, not fatal
    idx = tmp_path / "vc" / "experience" / "index.jsonl"
    idx.write_text(idx.read_text() + '{"id": "broken", "kind": "heard"', encoding="utf-8")
    assert VoiceCore(tmp_path / "vc").buffer.stats()["heard_utterances"] == 1


def _server_with(service):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_audit_fixes(tmp_path):
    import http.client

    cfg = VoiceCoreConfig.tiny(); cfg.max_response_seconds = 0.3
    engine = VoiceCore(tmp_path / "vc", cfg)
    service = EngineService(engine, max_input_seconds=1.0)
    httpd, base = _server_with(service)
    try:
        wav = audio_to_wav_bytes(proto_speech(0.8, seed=2), 16000)
        # cluster names: control characters rejected, any script accepted and percent-encoded in the header
        _post(base + "/api/hear", wav)
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(base + "/api/name?cluster=0&label=" + urllib.parse.quote("evil\r\nX-Injected: 1"))
        assert exc.value.code == 400
        named = json.loads(_post(base + "/api/name?cluster=0&label=" + urllib.parse.quote("हिन्दी <b>")).read())
        assert named["clusters"][0]["name"] == "हिन्दी <b>"
        r = _post(base + "/api/respond?learn=1", wav)
        header = r.headers["X-Cluster-Name"]
        assert header.isascii() and urllib.parse.unquote(header) == "हिन्दी <b>"
        r.read()
        # cross-site requests are refused, same-origin and non-browser requests pass
        req = urllib.request.Request(base + "/api/refit?k=1", data=b"", method="POST", headers={"Origin": "https://evil.example"})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 403
        req = urllib.request.Request(base + "/api/refit?k=1", data=b"", method="POST", headers={"Sec-Fetch-Site": "cross-site"})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 403
        req = urllib.request.Request(base + "/api/refit?k=1", data=b"", method="POST", headers={"Origin": base, "Sec-Fetch-Site": "same-origin"})
        assert urllib.request.urlopen(req, timeout=10).status == 200
        # a truncated final PCM frame (odd byte count) is dropped and the upload still works (never a 500)
        y_odd, _ = wav_bytes_to_audio(_post(base + "/api/roundtrip", wav[:-1]).read())
        assert len(y_odd) > 0
        # a long upload is decoded only up to max_input_seconds (tail), so the answer is bounded
        long_wav = audio_to_wav_bytes(proto_speech(6.0, seed=3), 16000)
        y, _ = wav_bytes_to_audio(long_wav, max_seconds=1.0)
        assert abs(len(y) - 16000) <= 1
        y2, _ = wav_bytes_to_audio(urllib.request.urlopen(urllib.request.Request(
            base + "/api/roundtrip", data=long_wav, method="POST", headers={"Content-Type": "audio/wav"}), timeout=60).read())
        assert abs(len(y2) - 16000) <= 320
        # a GET with a body does not desync the keep-alive connection
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
        conn.request("GET", "/healthz", body=b"x" * 100, headers={"Content-Length": "100"})
        assert conn.getresponse().read() and True
        conn.request("GET", "/healthz")
        assert json.loads(conn.getresponse().read())["ok"]
        conn.close()
        # a corrupt stored clip is skipped, status keeps working
        clip = next((tmp_path / "vc" / "experience" / "heard").glob("*.wav"))
        clip.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
        assert json.loads(urllib.request.urlopen(base + "/api/status", timeout=10).read())["experience"]["heard_utterances"] >= 1
        assert engine.buffer.heard_clips() is not None
        # generic 500 body: internal exception text never reaches the client
        original = service.roundtrip
        service.roundtrip = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret path /etc/x"))
        try:
            with pytest.raises(urllib.error.HTTPError) as exc:
                _post(base + "/api/roundtrip", wav)
            assert exc.value.code == 500 and "secret" not in exc.value.read().decode()
        finally:
            service.roundtrip = original
    finally:
        httpd.shutdown()
    # non-ASCII Authorization header is a clean 401 when a token is set; config.json is not rewritten on open
    service2 = EngineService(engine, token="s3cret-token")
    httpd2, base2 = _server_with(service2)
    try:
        req = urllib.request.Request(base2 + "/api/status", headers={"Authorization": "Bearer café"})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 401
    finally:
        httpd2.shutdown()
    with pytest.raises(ValueError):
        EngineService(engine, token="short")
    cfg_path = tmp_path / "vc" / "config.json"
    before = cfg_path.stat().st_mtime_ns
    VoiceCore(tmp_path / "vc")
    assert cfg_path.stat().st_mtime_ns == before
    assert not list((tmp_path / "vc").glob("*.tmp"))


def test_selftrain_stops_cooperatively(tmp_path):
    from voicecore.selftrain.loop import SelfTrainer

    cfg = VoiceCoreConfig.tiny()
    sc = cfg.selftrain
    sc.min_buffer_seconds = 1.0; sc.codec_snr_target_db = -100; sc.codec_steps_per_round = 500; sc.lm_steps_per_round = 500
    engine = VoiceCore(tmp_path / "vc", cfg)
    engine.hear(proto_speech(2.0, seed=1))
    stop = threading.Event(); stop.set()
    t0 = __import__("time").time()
    rep = SelfTrainer(engine, stop_event=stop).round()
    assert "stopped" in rep["note"] and __import__("time").time() - t0 < 20


def test_serve_refuses_bad_settings(tmp_path, monkeypatch):
    from voicecore.server import serve
    cfg = VoiceCoreConfig.tiny()
    VoiceCore(tmp_path / "vc", cfg).save()
    monkeypatch.delenv("VOICECORE_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        serve(tmp_path / "vc", host="0.0.0.0", port=0)            # non-local without a token
    with pytest.raises(SystemExit):
        serve(tmp_path / "new", host="127.0.0.1", port=0, init_preset="nope")
    assert not (tmp_path / "new" / "config.json").exists()


def test_engine_lock_and_unique_temp_files(tmp_path):
    from voicecore.fsutil import EngineLock, atomic_write_text
    root = tmp_path / "vc"; root.mkdir()
    with EngineLock(root):
        with pytest.raises(RuntimeError):
            EngineLock(root).acquire()
    EngineLock(root).acquire().release()                          # released cleanly
    # two writers of the same file from different threads never collide on a temp name
    errors = []
    def write(i):
        try:
            for _ in range(20):
                atomic_write_text(root / "state.json", json.dumps({"writer": i}))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    ts = [threading.Thread(target=write, args=(i,)) for i in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errors and json.loads((root / "state.json").read_text())["writer"] in range(4)
    assert not list(root.glob("*.tmp"))
