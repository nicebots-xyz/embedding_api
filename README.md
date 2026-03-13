# embedding-api

A lightweight HTTP API that generates image embeddings using [OpenCLIP](https://github.com/mlfoundations/open_clip)'s `ViT-B-16-SigLIP-256` model (pretrained on WebLI). Runs entirely on CPU.

## Endpoints

### `POST /embed/upload`
Upload an image file directly.

```bash
curl -X POST http://localhost:8000/embed/upload \
  -F "file=@image.jpg"
```

### `POST /embed/url`
Fetch and embed an image from a public URL.

```bash
curl -X POST http://localhost:8000/embed/url \
  -F "url=https://example.com/image.jpg"
```

Both endpoints return:
```json
{
  "embedding": [0.123, ...],
  "model": "ViT-B-16-SigLIP-256/webli",
  "dimensions": 512
}
```

## Running

### Docker Compose (dev)
```bash
docker compose -f compose.dev.yml up --build
```
API will be available at `http://localhost:6789`.

## Configuration

| Variable        | Default | Description                              |
|-----------------|---------|------------------------------------------|
| `MAX_CONCURRENT`| `4`     | Max simultaneous inference requests      |

Interactive docs are available at `/docs`.