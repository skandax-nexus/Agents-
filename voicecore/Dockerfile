# VoiceCore: from-scratch voice-to-voice engine with its local web UI.
#   docker build -t voicecore .
#   docker run --rm -p 127.0.0.1:8765:8765 -v voicecore-engine:/data -e VOICECORE_TOKEN=change-me-please voicecore
# VOICECORE_TOKEN is required: the container binds 0.0.0.0 and every /api/* call must carry it.
# The engine is created in the /data volume on first start (preset from the command line, default cpu).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home --uid 1000 voicecore
WORKDIR /app
COPY pyproject.toml README.md ./
COPY voicecore ./voicecore
COPY scripts ./scripts
# CPU-only torch keeps the image small; swap the index for a CUDA build if you have a GPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.*" && pip install . && mkdir -p /data && chown -R voicecore /data
USER voicecore
VOLUME ["/data"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=3).status == 200 else 1)"
# exec form: the python process is PID 1 and receives SIGTERM directly; no shell, no swallowed errors.
ENTRYPOINT ["voicecore", "serve", "/data/engine", "--host", "0.0.0.0", "--port", "8765"]
CMD ["--init-preset", "cpu"]
