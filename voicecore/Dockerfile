# VoiceCore: from-scratch voice-to-voice engine with its local web UI.
#   docker build -t voicecore .
#   docker run --rm -p 8765:8765 -v voicecore-engine:/data voicecore
# The engine lives in the /data volume; set VOICECORE_TOKEN to require a bearer token on the API.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home --uid 1000 voicecore
WORKDIR /app
COPY pyproject.toml README.md ./
COPY voicecore ./voicecore
COPY scripts ./scripts
# CPU-only torch keeps the image small; swap the index for a CUDA build if you have a GPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -e . && mkdir -p /data && chown -R voicecore /data
USER voicecore
VOLUME ["/data"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=3).status == 200 else 1)"
ENTRYPOINT ["/bin/sh", "-c", "voicecore init /data/engine --preset ${VOICECORE_PRESET:-cpu} 2>/dev/null || true; exec voicecore serve /data/engine --host 0.0.0.0 --port 8765 --token \"${VOICECORE_TOKEN:?set VOICECORE_TOKEN}\" --log-level ${VOICECORE_LOG_LEVEL:-info}"]
