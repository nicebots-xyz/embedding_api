FROM python:3.11-slim@sha256:b27df5841f3355e9473f9a516d38a6783b6c8dfeacaf2d14a240f443b368ddb6 AS deps

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pdm
ENV PDM_CHECK_UPDATE=false

COPY pyproject.toml pdm.lock LICENSE /app/

RUN pdm install --frozen-lockfile --prod --no-editable --no-self

FROM python:3.11-slim@sha256:b27df5841f3355e9473f9a516d38a6783b6c8dfeacaf2d14a240f443b368ddb6 AS model-download

COPY --from=deps /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    TORCH_HOME=/model-cache \
    HF_HOME=/model-cache/huggingface

RUN python - <<'EOF'
import open_clip
open_clip.create_model_and_transforms(
    "ViT-B-16-SigLIP-256",
    pretrained="webli",
    device="cpu",
)
print("Model weights cached.")
EOF

FROM python:3.11-slim@sha256:b27df5841f3355e9473f9a516d38a6783b6c8dfeacaf2d14a240f443b368ddb6 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --no-create-home --uid 1001 appuser

COPY --from=deps /app/.venv /app/.venv
COPY --chown=appuser --from=model-download /model-cache /model-cache

WORKDIR /app
COPY main.py .
RUN chown -R appuser /app

USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    TORCH_HOME=/model-cache \
    HF_HOME=/model-cache/huggingface \
    MAX_CONCURRENT=4 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
