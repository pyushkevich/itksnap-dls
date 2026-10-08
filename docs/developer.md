# Developer Guide

This guide is for people who want to modify `itksnap-dls` — add a model, add an endpoint,
or debug the server. For installing and using it, see the [Quick Start](quick_start.md).
For how to submit a change, see
[CONTRIBUTING.md](https://github.com/pyushkevich/itksnap-dls/blob/main/CONTRIBUTING.md).

## What this package is

`itksnap-dls` is a REST API server that lets ITK-SNAP perform interactive, AI-assisted
segmentation. ITK-SNAP sends an image and a stream of user interactions — clicks,
scribbles, lasso polygons — over HTTP, and the server returns a binary segmentation mask
after each one. Because it is a plain HTTP service, the model can run on a GPU machine
somewhere else entirely, which is the point: ITK-SNAP itself has no deep-learning
dependencies.

| Concern            | Library                                                    |
| ------------------ | ---------------------------------------------------------- |
| Web framework      | FastAPI + Uvicorn (ASGI)                                    |
| Deep learning      | PyTorch, with nnInteractive and SAM 2 as the wrapped models |
| Model distribution | Hugging Face Hub                                            |
| Medical image I/O  | SimpleITK                                                   |
| Public tunneling   | ngrok (optional)                                            |

## Repository layout

```
itksnap-dls/
├── itksnap_dls/
│   ├── __init__.py     # re-exports the FastAPI `app`
│   ├── __main__.py     # CLI entry point: python -m itksnap_dls
│   ├── server.py       # the FastAPI app and every route handler
│   ├── segment.py      # configuration and the model wrappers
│   └── session.py      # the in-memory session registry
├── docs/               # Sphinx sources (this guide lives here)
├── tests/
├── .github/workflows/python-publish.yml
├── pyproject.toml
└── LICENSE.txt
```

## Architecture

There are three pieces, and the separation between them is what makes adding a model easy:

**`server.py` owns HTTP.** It defines routes, decodes and encodes payloads, and looks up
sessions. It does not know how any particular model works — it only calls the
`ModelWrapper` interface.

**`segment.py` owns models.** Each supported model is a `ModelWrapper` subclass that
adapts that model to a single common interface. This is the layer to extend.

**`session.py` owns state.** A `SessionManager` singleton maps a session ID to the live
`ModelWrapper` instance holding that client's image and interaction state.

```
ITK-SNAP  ──HTTP──▶  server.py  ──▶  session.py  ──▶  ModelWrapper (segment.py)  ──▶  PyTorch
```

### `__main__.py` — entry point

Parses CLI arguments, populates the `global_config` object imported from `segment.py`,
optionally opens an ngrok tunnel, prints a connection banner listing the reachable
addresses, and starts Uvicorn.

| Flag                   | Default                            | Description                                          |
| ---------------------- | ---------------------------------- | ---------------------------------------------------- |
| `--port`, `-p`         | `8911`                             | TCP port                                              |
| `--host`, `-H`         | `0.0.0.0`                          | Bind address                                          |
| `--models-path`, `-m`  | Hugging Face default               | Directory for the model cache                         |
| `--device`             | `cuda` if available, else `cpu`    | Torch device: `cpu`, `cuda`, or `mps`                 |
| `-k`, `--insecure`     | off                                | Skip HTTPS certificate verification                   |
| `--no-network`         | off                                | Do not contact external networks to download models   |
| `-N`, `--ngrok`        | off                                | Open a public ngrok tunnel; needs `NGROK_AUTHTOKEN`   |
| `--use-colors`         | off                                | Force colored terminal output                         |
| `--setup-only`         | off                                | Download models and exit without serving              |
| `--mock-models`        | off                                | Replace models with fast CPU stand-ins for testing    |

`--setup-only` exists so that a machine can pre-fetch weights during provisioning rather
than on a user's first click.

`--mock-models` swaps each model for a stand-in with the same ID and capabilities that does
no inference: point clicks paint a small ball, scribbles are dilated, and lassos are used
as-is. This makes it possible to test the server and the ITK-SNAP client on a machine
without a GPU.

### `segment.py` — configuration and models

**`SegmentServerConfig`** holds runtime settings — `hf_models_path`, `device`,
`n_cpu_threads`, `https_verify`, `https_enabled`. A single module-level instance,
`global_config`, is populated by `__main__.py` at startup and used as the default argument
to every wrapper constructor.

**`config_hf_backend()`** configures the Hugging Face HTTP client, handling both the
`configure_http_backend` and `set_client_factory` APIs depending on the installed version,
and disabling certificate verification when `--insecure` was passed.

**`ModelWrapper`** is the interface every model implements. The class attributes declare
what the model can do, and are what `/v2/models` reports to the client:

| Attribute      | Meaning                                                            |
| -------------- | ------------------------------------------------------------------ |
| `ID`           | Identifier used in URLs and in the model listing                   |
| `DIMENSIONS`   | `2` or `3`                                                          |
| `CHANNELS`     | Supported channel counts; an empty list means any                  |
| `INTERACTIONS` | Interaction types supported, e.g. `["point", "box", "scribble", "lasso"]` |

The instance methods form the segmentation lifecycle:

| Method                                              | Purpose                                      |
| --------------------------------------------------- | -------------------------------------------- |
| `set_image(sitk_image)`                             | Load the image and prepare inference state   |
| `add_point_interaction(index_itk, include_interaction)` | Apply a click                             |
| `add_scribble_interaction(sitk_image, include_interaction)` | Apply a scribble mask                 |
| `add_lasso_interaction(sitk_image, include_interaction)`    | Apply a lasso mask                    |
| `reset_interactions()`                              | Clear prompts, keep the loaded image         |
| `get_result()`                                      | Return the current mask as a SimpleITK image |

`include_interaction` distinguishes a positive prompt (include this region) from a negative
one (exclude it).

Two wrappers ship today: **`nnInteractiveWrapper`** (3D, single-channel, all four
interaction types) and **`SAM2Wrapper`** (point interactions).

**`get_model_listing()`** builds the `/v2/models` response by reading the class attributes
of each registered wrapper. **`instantiate_model_wrapper(repo_id, config)`** maps an `ID`
back to a constructed wrapper.

### `session.py` — session state

`SessionManager` is a thin dictionary wrapper — `create_session`, `get_session`,
`delete_session` — with a module-level singleton, `session_manager`. A session's value is
the live `ModelWrapper`, which is holding the image on the GPU.

Two consequences worth knowing:

- **Sessions are in-memory and per-process.** Restarting the server drops every session,
  and the server cannot be scaled across multiple worker processes without a shared store.
- **Sessions are freed only when the client calls `/end_session`.** A client that
  disconnects without ending its session leaves the image resident in GPU memory. If you
  are debugging "the GPU filled up," this is the first thing to check.

### `server.py` — HTTP layer

Defines `app`, the FastAPI instance re-exported from `__init__.py`.

Every interaction handler follows the same shape, which is worth recognizing before you
add one:

1. Look up the session; return `{"error": "Invalid session"}` if it is unknown.
2. Decode any uploaded image.
3. Call the corresponding `ModelWrapper` method.
4. Threshold the result to a binary `int8` array, gzip it, base64-encode it, and return it
   as `{"status": "success", "result": <base64>}`.
5. Log timings for the inference and encoding steps.

**Image transfer format.** Images move as gzip-compressed raw `float32` buffers plus a JSON
`metadata` form field. `read_sitk_image()` decompresses the buffer, reshapes it using
`metadata["dimensions"]` **reversed** — SimpleITK arrays are indexed as
`(z, y, x)` while the metadata lists dimensions in `(x, y, z)` order — and builds a vector
image when `components_per_pixel` is greater than one. Getting this ordering wrong produces
a transposed image rather than an error, so it is worth being careful here.

Results travel the other way as gzip-compressed base64-encoded binary masks.

## REST API

| Method | Route                                         | Purpose                                          |
| ------ | --------------------------------------------- | ------------------------------------------------ |
| GET    | `/status`                                     | Health check; returns the package version        |
| GET    | `/v2/models`                                  | List available models and their capabilities     |
| GET    | `/v2/start_session/{model_id}`                | Start a session with a named model               |
| GET    | `/start_session`                              | Legacy: start a session with nnInteractive       |
| POST   | `/v2/upload_raw/{session_id}`                 | Upload the image to segment                      |
| GET    | `/v2/process_point_interaction/{session_id}`  | Apply a click (`point=`, repeated; `foreground=`) |
| GET    | `/process_point_interaction/{session_id}`     | Legacy: same, with separate `x=`, `y=`, `z=`     |
| POST   | `/process_scribble_interaction/{session_id}`  | Apply a scribble mask                            |
| POST   | `/process_lasso_interaction/{session_id}`     | Apply a lasso mask                               |
| GET    | `/v2/reset_interactions/{session_id}`         | Clear prompts, keep the image                    |
| GET    | `/v2/end_session/{session_id}`                | End the session and free its memory              |

FastAPI serves interactive documentation for all of this at `http://<host>:<port>/docs` —
the fastest way to try a route by hand.

**The unprefixed routes are a compatibility surface, not duplicates to clean up.**
Released versions of ITK-SNAP call them. Several are registered by stacking two decorators
on one handler, so the `/v2` and legacy paths stay in step automatically. Removing them
breaks installed clients.

## Development setup

```bash
git clone https://github.com/pyushkevich/itksnap-dls.git
cd itksnap-dls
python -m venv .venv && source .venv/bin/activate    # Python 3.10 or newer
pip install -e ".[test]"

python -m itksnap_dls --port 8911                    # run the server
python -m itksnap_dls --setup-only                   # just download models
python -m itksnap_dls --mock-models                  # run without GPU/models
pytest                                               # fast tests
```

Building these docs:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Adding a model

The model layer is designed so that this touches one file plus one line elsewhere.

**1. Write the wrapper** in `segment.py`, declaring its capabilities and implementing the
lifecycle methods:

```python
class MyModelWrapper(ModelWrapper):
    HF_REPO_ID = "my-org/my-model"

    ID = "MyModel"
    DIMENSIONS = 3
    CHANNELS = [1]
    INTERACTIONS = ["point", "scribble"]

    def __init__(self, config: SegmentServerConfig = global_config):
        config_hf_backend()
        # download weights via huggingface_hub, honoring config.hf_models_path,
        # and move the model to config.device
        ...

    def set_image(self, sitk_image):
        ...

    def add_point_interaction(self, index_itk, include_interaction):
        ...

    def get_result(self) -> sitk.Image:
        ...
```

Only implement the interaction methods you list in `INTERACTIONS`.

**2. Register it** in the two places `segment.py` enumerates models — the `models` list in
`get_model_listing()` and the branch in `instantiate_model_wrapper()`.

**3. Declare heavy dependencies as an optional extra** in `pyproject.toml`, following the
`totalseg` extra, rather than adding them to the base dependencies. The base install
already pulls in PyTorch and is large.

**4. Add a test**, with an `integration_<model>` marker registered in `pyproject.toml` if
it needs real weights.

No change to `server.py` is required: routes dispatch through `ModelWrapper`, and
`/v2/models` picks up the new entry from the class attributes.

## Adding an endpoint

Add the handler to `server.py`, following the shape described above. For a new interaction
type, also add the corresponding method to `ModelWrapper` and to every wrapper that
declares support for it, and list the type in that wrapper's `INTERACTIONS`.

Prefix genuinely new routes with `/v2/`. Do not change the meaning of an existing route —
add a new one and let old clients keep using the old path.

## Conventions

- **The wrapper interface is the extension point.** Model-specific behavior belongs in
  `segment.py`; if you find yourself adding an `if model_id == ...` branch to `server.py`,
  the capability probably belongs on `ModelWrapper` instead.
- **Configuration flows through `global_config`,** populated once at startup. Wrappers take
  it as a defaulted constructor argument, which keeps them testable with a custom config.
- **Return errors as JSON,** matching the existing `{"error": ...}` shape, so the ITK-SNAP
  client can present them.
- **Log timings** around inference and encoding. The existing handlers do this, and it is
  what makes remote performance problems diagnosable.

## Releasing

Version is set in `pyproject.toml`. Publication to PyPI is automated by
`.github/workflows/python-publish.yml`. Because ITK-SNAP clients in the field talk to
whatever server version a user installed, bump the version on any API-visible change and
note it in the release.
