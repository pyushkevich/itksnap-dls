# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`itksnap-dls` is a FastAPI-based HTTP server that bridges [ITK-SNAP](https://itksnap.org) (a medical image segmentation GUI) to GPU-accelerated deep learning models. ITK-SNAP connects over HTTP to this server, uploads image slabs, sends user interactions (point clicks, scribbles, lasso polygons), and receives back binary segmentation masks.

The two supported models are:
- **nnInteractive** (`MIC-DKFZ/nnInteractive` on HuggingFace) — 3D volumetric segmentation; primary model
- **SAM2** (`facebook/sam2.1-hiera-large`) — 2D slice-based segmentation

## Running the server

```bash
# Standard run (auto-detects GPU)
python -m itksnap_dls

# Typical dev invocation (matches VSCode launch config)
python -m itksnap_dls -k --no-network

# Key CLI flags
python -m itksnap_dls --port 8911 --device cuda --models-path /path/to/models
python -m itksnap_dls -k          # skip HTTPS cert verification (needed on many systems)
python -m itksnap_dls --no-network  # don't attempt to download models from HuggingFace
python -m itksnap_dls --setup-only  # download models then exit (no server)
python -m itksnap_dls --mock-models  # fast CPU stand-ins instead of real models (for testing)
python -m itksnap_dls -N           # expose via ngrok (requires NGROK_AUTHTOKEN env var)
```

Default port is **8911**. The server prints all accessible URLs at startup.

## Installing for development

```bash
pip install -e .
```

## Building for PyPI

```bash
python -m build
```

Publishing to PyPI is automated by `.github/workflows/python-publish.yml`, which runs when a GitHub release is published and can also be started manually (`gh workflow run python-publish.yml`). Bump `version` in `pyproject.toml` first: PyPI rejects re-uploads of an existing version, so a second run for the same version fails at the publish step.

## Code architecture

All server code lives in `itksnap_dls/`:

| File | Role |
|------|------|
| `__main__.py` | CLI entry point; argument parsing, GPU detection, ngrok setup, uvicorn launch |
| `server.py` | FastAPI app; all HTTP endpoints |
| `segment.py` | Model wrappers (`nnInteractiveWrapper`, `SAM2Wrapper`), their mock stand-ins (`MockModelWrapper` subclasses), and `SegmentServerConfig` |
| `session.py` | In-memory `SessionManager` keyed by UUID string |

**Request flow:**
1. Client calls `GET /v2/start_session/{model_id}` → server instantiates model, returns `session_id`
2. Client calls `POST /v2/upload_raw/{session_id}` with a gzip-compressed float32 array + JSON metadata → server decodes to a SimpleITK image and calls `seg.set_image()`
3. Client calls `GET /v2/process_point_interaction/{session_id}?point=x&point=y&point=z&foreground=true` (or scribble/lasso POST variants) → server runs inference, returns base64-encoded gzip-compressed int8 segmentation mask
4. Client calls `GET /v2/end_session/{session_id}` when done

**Legacy (v1) endpoints** without `/v2/` prefix are thin wrappers kept for backwards compatibility with older ITK-SNAP builds.

**Image encoding convention:** Images are transmitted as raw float32 bytes, gzip-compressed, with a separate JSON `metadata` field containing `dimensions` (list, x-y-z order) and `components_per_pixel`. ITK index ordering is reversed relative to numpy array ordering — `nnInteractiveWrapper.add_point_interaction` handles this with `index_itk[::-1]`.

**`SegmentServerConfig`** is a global singleton (`global_config` in `segment.py`) mutated by `__main__.py` before models are loaded. It controls device, models cache path, HTTPS behavior, and `mock_models`.

**Mock models:** with `--mock-models`, `get_model_classes()` returns `nnInteractiveMockWrapper` and `SAM2MockWrapper` instead of the real wrappers. They advertise the same IDs and capabilities but perform no inference (points paint a ball, scribbles are dilated, lassos are used as-is), so the server and ITK-SNAP client can be tested on CPU without downloading models. Both `/v2/models` and `start_session` go through `get_model_classes()`.

## API reference

`GET /status` — health check, returns version  
`GET /v2/models` — list models with their capabilities (dimensions, channels, interaction types). Note: nnInteractive advertises `"box"`, but there is no box interaction endpoint yet  
`GET /v2/start_session/{model_id}` — `model_id` is `"nnInteractive"` or `"SAM2"`  
`POST /v2/upload_raw/{session_id}` — multipart: `file` (gzipped float32) + `metadata` (JSON string)  
`GET /v2/process_point_interaction/{session_id}` — query params: `point` (repeatable int), `foreground` (bool)  
`POST /process_scribble_interaction/{session_id}` — same multipart format as upload_raw, plus `foreground`  
`POST /process_lasso_interaction/{session_id}` — same  
`GET /v2/reset_interactions/{session_id}` — clears all interactions for a session  
`GET /v2/end_session/{session_id}` — frees the session
