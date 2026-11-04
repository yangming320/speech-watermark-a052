FROM python:3.12-slim
ARG TORCH_VERSION=2.14.1
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/speechwm-cache/matplotlib XDG_CACHE_HOME=/tmp/speechwm-cache OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-lock.txt /app/requirements-lock.txt
RUN python -m pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
RUN python -c "from pathlib import Path;p=Path('requirements-lock.txt');Path('/tmp/requirements-nontorch.txt').write_text('\n'.join(l for l in p.read_text().splitlines() if not l.startswith('torch=='))+'\n')" && python -m pip install --no-cache-dir -r /tmp/requirements-nontorch.txt
COPY pyproject.toml /app/
COPY src /app/src
COPY deployment /app/deployment
COPY scripts/preflight_demo.py /app/scripts/preflight_demo.py
RUN python -m pip install --no-cache-dir . --no-deps && useradd --create-home --uid 10001 speechwm && mkdir -p /tmp/speechwm-cache && chown -R speechwm:speechwm /tmp/speechwm-cache
USER speechwm
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/api/info',timeout=3)"
CMD ["python", "-m", "speechwm.demo", "--root", "/app", "--manifest", "deployment/models.json", "--host", "0.0.0.0", "--port", "7860"]
