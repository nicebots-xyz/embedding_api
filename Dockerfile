FROM python:3.11-slim AS deps

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project \
        --python python3.11 \
        --compile-bytecode

FROM python:3.11-slim AS model-download

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

FROM python:3.11-slim AS runtime

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
