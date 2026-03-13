import asyncio
import io
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
import open_clip
import torch
import torch.nn.functional as F  # noqa: N812
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, HttpUrl

MODEL_NAME = "ViT-B-16-SigLIP-256"
PRETRAINED = "webli"

MAX_CONCURRENT: int = int(os.environ.get("MAX_CONCURRENT", "4"))

_model: open_clip.CLIP
_preprocess: object
_semaphore: asyncio.Semaphore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    global _model, _preprocess, _semaphore  # noqa: PLW0603
    _model, _, _preprocess = open_clip.create_model_and_transforms(  # pyright: ignore[reportAssignmentType]
        MODEL_NAME,
        pretrained=PRETRAINED,
        device="cpu",
        precision="fp32",
    )
    _model.eval()
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    yield


app = FastAPI(title="Image Embedding API", lifespan=lifespan)


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    model: str
    dimensions: int


def _embed_sync(image: Image.Image) -> list[float]:
    """CPU inference — runs in a thread-pool executor to avoid blocking the event loop."""
    tensor: torch.Tensor = _preprocess(  # pyright: ignore[reportUnknownVariableType, reportCallIssue]
        image
    ).unsqueeze(0)
    with torch.no_grad(), torch.autocast("cpu"):
        features = _model.encode_image(tensor)
        features = F.normalize(features, dim=-1)
    return features[0].tolist()  # pyright: ignore[reportUnknownVariableType]


async def _embed(image: Image.Image) -> list[float]:
    """Acquire a concurrency slot, then run inference off the event loop."""
    async with _semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _embed_sync, image)


def _build_response(embedding: list[float]) -> EmbeddingResponse:
    return EmbeddingResponse(
        embedding=embedding,
        model=f"{MODEL_NAME}/{PRETRAINED}",
        dimensions=len(embedding),
    )


@app.post("/embed/upload")
async def embed_upload(file: Annotated[UploadFile, File(description="Image file")]) -> EmbeddingResponse:
    """Compute image embedding from a multipart file upload."""
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail="Could not decode image.") from e
    return _build_response(await _embed(image))


@app.post("/embed/url")
async def embed_url(url: Annotated[HttpUrl, Form(description="Publicly accessible image URL")]) -> EmbeddingResponse:
    """Compute image embedding by downloading an image from a URL."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        try:
            resp = await client.get(str(url))
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=422, detail="Failed to fetch image") from e
    try:
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail="Could not decode image from URL.") from e
    return _build_response(await _embed(image))


def main() -> None:
    import uvicorn  # noqa: PLC0415

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
