# LLM Serve Dashboard

A single-file, dependency-free live dashboard for your local LLM serving box — GPU utilization,
per-model throughput, KV/context fill, and system stats for **llama.cpp** and **vLLM**, in one
green terminal-styled page.

No framework, no build step, no external requests. The frontend is one `index.html`; the backend is
one stdlib Python file that reads `nvidia-smi` and each server's Prometheus `/metrics` — and also
serves the page, so the dashboard is same-origin and needs no CORS grant.

![LLM Serve Dashboard](docs/screenshot.png)

*(Screenshot rendered from the committed `docs/example-metrics.json` fixture — no real hardware,
model names or network data. Plug in your own box and it goes live.)*

Regenerate it yourself, so the caption above is checkable rather than something you take on trust:

```bash
FLEET_METRICS_FIXTURE=docs/example-metrics.json python3 fleet-metrics.py &
chromium --headless --window-size=2880,1640 --virtual-time-budget=9000 \
         --screenshot=docs/screenshot.png http://localhost:8092/
```

## What it shows

- **GPUs** — per-card utilization, VRAM, power, temperature, clocks, and the actual compute
  tenants on each card (pulled from `nvidia-smi --query-compute-apps`, so cards are labeled from
  ground truth, not VRAM guesswork).
- **Primary worker** — decode & prefill tokens/sec, request counts, context/KV fill. Works with
  llama.cpp (`/metrics` + `/props`) and vLLM (`/metrics` + `/v1/models`). The worker port is
  **auto-discovered** from listening sockets, so a bench or swap that moves the model to another
  port still lands on the dashboard.
- **Secondary servers** — an optional row of cards for extra llama-servers running **on this same
  machine**, each on its own port (a small CPU model, a second engine, etc.). LoRA adapters are
  shown per secondary card. Configure with `SECONDARY_SERVERS`.
- **System** — RAM, load average, and per-NIC network rates.
- **Model library** — a browsable inventory of your loadable models with quant, ctx, and measured
  throughput, driven by a JSON registry you edit.
- **Reasoning tap (optional)** — off by default. Point `THOUGHT_LOG` at a log of streamed
  `reasoning_content` and the panel shows its live tail. Separate per-request CoT panels require a
  proxy serving structured streams at `/thoughts` on `:8090`; that proxy is **not** included here.

## Run it

Requirements: Python 3.8+ and `nvidia-smi` (NVIDIA GPUs). No pip installs.

```bash
# 1. start the backend (serves the page AND the JSON on :8092)
python3 fleet-metrics.py

# 2. open the dashboard
xdg-open http://localhost:8092/
```

The page polls `/metrics` on its own origin for live data. That's it.

> Opening `index.html` directly from `file://` no longer works, deliberately. A `file://` page has
> `Origin: null`, and so does every sandboxed iframe — granting it would let any page that can
> embed one read your telemetry. Serving the page from the backend makes it same-origin instead.

### Point it at your servers

By default it looks for a llama.cpp/vLLM server on `:8001` (then `:8010`, `:8123`). Override:

```bash
# candidate ports for the primary worker.
# :8001 wins outright if it answers; otherwise the responder with the largest ctx wins.
WORKER_PORT_CANDIDATES=8000,8001,8080 python3 fleet-metrics.py

# secondary servers shown as extra cards: label:port,label:port
# The label is only a display name — every port is fetched on THIS machine (localhost).
# There is no remote-host support; pointing a label at another box will not reach it.
SECONDARY_SERVERS='cpu-a:9093,cpu-b:9095' python3 fleet-metrics.py

# model library source (defaults to the bundled models-registry.json)
MODELS_REGISTRY=~/my-models.json python3 fleet-metrics.py

# optional reasoning/CoT panel: tail a log of streamed reasoning_content
THOUGHT_LOG=~/thinking.log python3 fleet-metrics.py
```

Also update the matching `SECONDARY_META` list near the bottom of `index.html` so the secondary
cards render with your names.

## How it works

`fleet-metrics.py` is a tiny `http.server` on `:8092` exposing:

- `GET /` — the dashboard page itself, so it loads same-origin.
- `GET /metrics` — the full JSON blob the dashboard renders (GPUs, worker, secondaries, system).
- `GET /models` — the model-library registry.
- `GET /health` — liveness.

llama.cpp publishes ready-made rate gauges (`prompt_tokens_seconds`, `predicted_tokens_seconds`)
alongside its cumulative counters, and those gauges are read directly — note they read 0 while the
model is idle between generations. vLLM has no such gauges, so its rates are derived from deltas
between cumulative counters across polls. All parsing is stdlib.

## Python or Rust? Two builds

This repo is the **Python build** — the zero-setup, read-the-whole-thing-in-one-file version. There
is also a **Rust build, [fleet-tap](https://github.com/NHClimber87/fleet-tap)**, which is a
transparent traffic *tap* rather than just a metrics scraper. They render the same dashboard; they
differ in how they get the numbers and what they cost to run.

**Use the Python build (this repo) when:**
- You want it running in ten seconds: `python3 fleet-metrics.py`, no toolchain, no compile step.
- You watch one box with one or a handful of servers.
- You'd rather read and hack a single stdlib file than a Rust crate.
- You don't want anything sitting in front of your serving ports — this only ever *reads* each
  server's `/metrics`, never proxies traffic.

**Use the Rust build ([fleet-tap](https://github.com/NHClimber87/fleet-tap)) when:**
- You want **accurate, client-measured throughput.** It measures real tokens/sec from the actual
  request/response stream it taps, instead of scraping llama.cpp's `predicted_tokens_seconds` gauge
  (which reads 0 whenever the model is idle between generations).
- You run **many endpoints** and want live per-endpoint traffic streaming, **SSE push** (the browser
  stops polling), and **on-disk retention/history**.
- You want a single compiled binary with hot-reloadable config (add an endpoint, no restart).
- You accept the tradeoff: fleet-tap works by **holding each canonical serving port and forwarding
  to the engine on `port+10000`**, so it's a bit more setup and it sits in the request path (a
  fleet-tap crash interrupts serving until it restarts). Fine on a box you fully control; more than
  a casual "just show me the GPUs" tool needs.

Short version: **Python for the quick, side-car glance at one box; Rust for the always-on tap on a
busy multi-model rig.**

## Notes

- **Local-first / no phone-home.** No CDNs, no web fonts, no analytics, no external requests of any
  kind. Every request the page makes goes to its own origin: `/metrics` on a 2-second poll, plus
  `/models` once for the model library.
- **Loopback by default.** The metrics server binds `127.0.0.1:8092`. `/metrics` reports GPU
  tenants, system stats, ARP neighbours and established connections, so it is not something to put
  on a network implicitly. To reach it from another machine, opt in and put your own auth or tunnel
  in front of it — it is read-only telemetry, but it is unauthenticated:

  ```bash
  FLEET_METRICS_BIND=0.0.0.0 python3 fleet-metrics.py
  ```

- **DNS-rebinding guard.** Requests are refused unless the `Host` header is a loopback name.
  This is separate from CORS on purpose: a page at `http://rebind.example:8092` whose DNS flips
  to `127.0.0.1` becomes *same-origin* with this server, so no CORS check is even consulted, and
  binding to loopback doesn't help because the request comes from your own browser. The `Host`
  header is what still gives it away. Reaching the dashboard by a real hostname? Name it:

  ```bash
  FLEET_METRICS_ALLOWED_HOSTS=box.lan python3 fleet-metrics.py
  ```

- **Same-origin by default; CORS is an allowlist, never `*`.** The backend serves the page, so the
  dashboard's own fetches need no CORS grant at all. For anything else the server reflects only
  `localhost`/`127.0.0.1`/`[::1]` origins and sends no CORS header to the rest. This matters even
  on loopback: with `Access-Control-Allow-Origin: *`, any website you happened to visit could read
  your box's metrics via XHR from your own browser — binding to loopback does not prevent that.
  Serving `index.html` from somewhere else? Add its origin:

  ```bash
  FLEET_METRICS_ALLOWED_ORIGINS=http://192.0.2.10:8080 python3 fleet-metrics.py
  ```

## License

MIT — see [LICENSE](LICENSE).
