# LLM Serve Dashboard

A single-file, dependency-free live dashboard for your local LLM serving box — GPU utilization,
per-model throughput, KV/context fill, and system stats for **llama.cpp** and **vLLM**, in one
green terminal-styled page.

No framework, no build step, no external requests. The frontend is one `index.html` (opens on
`file://`); the backend is one stdlib Python file that reads `nvidia-smi` and each server's
Prometheus `/metrics`.

> Dark green terminal aesthetic: a left nav, a GPU grid (per-card util/VRAM/power/temp/tenants),
> a live throughput panel (decode & prefill tok/s), context/KV fill, system + network stats, and a
> browsable model library. Add your own screenshot here once you've run it.

## What it shows

- **GPUs** — per-card utilization, VRAM, power, temperature, clocks, and the actual compute
  tenants on each card (pulled from `nvidia-smi --query-compute-apps`, so cards are labeled from
  ground truth, not VRAM guesswork).
- **Primary worker** — decode & prefill tokens/sec, request counts, context/KV fill, LoRA
  adapters. Works with llama.cpp (`/metrics` + `/props`) and vLLM (`/metrics` + `/v1/models`).
  The worker port is **auto-discovered** from listening sockets, so a bench or swap that moves the
  model to another port still lands on the dashboard.
- **Secondary servers** — an optional row of cards for extra CPU/GPU llama-servers you run (point
  them at any endpoints — a small CPU model, a second box, etc.). Configure with `SECONDARY_SERVERS`.
- **System** — CPU, RAM, load, network, disk.
- **Model library** — a browsable inventory of your loadable models with quant, ctx, and measured
  throughput, driven by a JSON registry you edit.
- **Reasoning tap (optional)** — live per-request reasoning/CoT panels, if you tee a model's
  `reasoning_content` to a log (see `THOUGHT_LOG` below). Off by default.

## Run it

Requirements: Python 3.8+ and `nvidia-smi` (NVIDIA GPUs). No pip installs.

```bash
# 1. start the metrics endpoint (serves JSON on :8092)
python3 fleet-metrics.py

# 2. open the dashboard
xdg-open index.html      # or just double-click it / open in a browser
```

The page polls `http://localhost:8092` for live data. That's it.

### Point it at your servers

By default it looks for a llama.cpp/vLLM server on `:8001` (then `:8010`, `:8123`). Override:

```bash
# candidate ports for the primary worker (first responder with the largest ctx wins)
WORKER_PORT_CANDIDATES=8000,8001,8080 python3 fleet-metrics.py

# secondary servers shown as extra cards: name:port,name:port
SECONDARY_SERVERS='cpu-a:9093,box2:9001' python3 fleet-metrics.py

# model library source (defaults to the bundled models-registry.json)
MODELS_REGISTRY=~/my-models.json python3 fleet-metrics.py

# optional reasoning/CoT panel: tail a log of streamed reasoning_content
THOUGHT_LOG=~/thinking.log python3 fleet-metrics.py
```

Also update the matching `DREAMER_META` list near the bottom of `index.html` so the secondary
cards render with your names.

## How it works

`fleet-metrics.py` is a tiny `http.server` on `:8092` exposing:

- `GET /metrics` — the full JSON blob the dashboard renders (GPUs, worker, secondaries, system).
- `GET /models` — the model-library registry.
- `GET /health` — liveness.

llama.cpp exposes Prometheus counters at `/metrics`; the server derives per-second rates from the
cumulative counters. vLLM's cumulative counters are handled the same way. All parsing is stdlib.

## Notes

- **Local-first / no phone-home.** No CDNs, no web fonts, no analytics. Everything is served from
  your box; the page makes exactly one request, to `localhost:8092`.
- The metrics server binds `0.0.0.0:8092` and sets permissive CORS so the `file://` page can read
  it. If you expose the box, put it behind your own auth/tunnel — it's read-only telemetry, but
  it's unauthenticated by design for local use.

## License

MIT — see [LICENSE](LICENSE).
