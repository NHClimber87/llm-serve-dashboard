# Install

From nothing to your own GPUs on screen. Should take about two minutes.

The [README](README.md) is the reference — every environment variable, every design decision,
every security note. This page is just the path.

---

## 1. What you need

| | |
|---|---|
| **Python 3.8 or newer** | `python3 --version`. Nothing to install — the backend is stdlib only. |
| **`nvidia-smi` on your PATH** | `nvidia-smi --version`. Ships with the NVIDIA driver. |
| **A browser** | Any current one. |

There is no `pip install`, no `npm install`, no build step, and no toolchain. If you have
Python and a working NVIDIA driver, you already have everything.

> **No NVIDIA GPU?** It still runs — the GPU panel is simply empty and everything else
> (worker throughput, network, system) works normally. It will not crash and it will not
> refuse to start.

## 2. Start it

```bash
python3 fleet-metrics.py
```

You should see one line:

```
fleet-metrics serving on 127.0.0.1:8092/metrics
```

Open <http://localhost:8092/> — or `xdg-open http://localhost:8092/` on Linux,
`open http://localhost:8092/` on macOS.

That is the whole install. The backend serves the page *and* the data, so there is no second
process, no web server to configure, and no CORS to grant.

> **Do not open `index.html` by double-clicking it.** A `file://` page cannot read the
> metrics, deliberately — see the README's note on `Origin: null`. Always go through
> `http://localhost:8092/`.

### Port 8092 already taken?

```bash
FLEET_METRICS_PORT=9100 python3 fleet-metrics.py
```

The page follows automatically; it polls whatever origin it was served from. If the port is
busy you get one line telling you so, not a stack trace.

## 3. Check it actually worked

Three things, in order. If all three pass, you are done.

```bash
# 1. the server is alive
curl -s http://localhost:8092/health

# 2. it can see your GPUs  (prints a number — 0 is fine if you have no NVIDIA card)
curl -s http://localhost:8092/metrics | python3 -c "import json,sys; print(len(json.load(sys.stdin)['gpus']), 'GPU(s)')"

# 3. it found your model server  (prints the port, or None)
curl -s http://localhost:8092/metrics | python3 -c "import json,sys; print('worker port:', json.load(sys.stdin).get('worker_port'))"
```

In the browser you should see GPU cards filling in within a couple of seconds — the page
polls every 2s. Tiles that depend on a model server stay blank until one is running, which
is expected, not a fault.

## 4. Point it at your model server

The dashboard looks for a llama.cpp or vLLM server on `:8001`, then `:8010`, then `:8123`.
If yours is elsewhere, say so:

```bash
WORKER_PORT_CANDIDATES=8000,8080 python3 fleet-metrics.py
```

`:8001` wins outright if it answers; otherwise the responder with the largest context wins.
Extra servers on the same box can be shown as their own cards:

```bash
SECONDARY_SERVERS='cpu-a:9093,cpu-b:9095' python3 fleet-metrics.py
```

All of these can be combined on one line. The full list — model registry, reasoning tap, LAN
binding, allowed hosts — is in the README.

## 5. Keep it running

For a quick look, `Ctrl-C` when you are done and that is that.

To leave it up, run it as a user service — no root, starts with your session:

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/llm-dashboard.service <<EOF
[Unit]
Description=LLM Serve Dashboard
After=network.target

[Service]
Type=simple
WorkingDirectory=$PWD
ExecStart=$(command -v python3) $PWD/fleet-metrics.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now llm-dashboard
systemctl --user status llm-dashboard --no-pager
```

`WorkingDirectory` matters — the backend serves `index.html` from its own directory, so do
not move the files after writing the unit. To pass options, add them as
`Environment=FLEET_METRICS_PORT=9100` lines under `[Service]`.

Survive logout without an active session:

```bash
sudo loginctl enable-linger "$USER"
```

## 6. When something is wrong

**Nothing at `localhost:8092`.** Check the process is up and what it bound to:

```bash
ss -ltnp | grep 8092
```

Empty means it is not running or it failed to start — run it in the foreground and read the
message.

**`address already in use`.** Something else has the port. Move yours with
`FLEET_METRICS_PORT=9100`, or find the occupant with `ss -ltnp | grep 8092`.

**GPU panel is empty.** Run `nvidia-smi` by hand. If that fails, the driver is the problem,
not this tool. If it works but the panel is still empty, check the backend's stderr — it
prints exactly which query failed and whether it fell back.

**GPU cards show, but no PCIe or throttle detail.** Your driver renamed those fields
(`clocks_throttle_reasons.*` became `clocks_event_reasons.*` in newer builds). The backend
notices, drops the optional fields, and keeps the rest of the panel alive rather than losing
everything. You will see one line on stderr saying so. Nothing else is affected.

**Worker tiles blank / `worker port: None`.** Nothing is answering on the candidate ports.
Confirm your server is up (`curl -s localhost:8001/v1/models`) and set
`WORKER_PORT_CANDIDATES` to the right port.

**Throughput reads `0.0 tok/s` with an age like `12s ago`.** Working as intended. Both
engines only publish rates while generating; a held value is labeled with its age and expires
after 60 seconds rather than lying about being current.

**Latency tile says vLLM-only.** Also intended. llama.cpp exports no latency histogram, so
nothing can scrape per-request latency from it — the tile says so instead of inventing a
number. If you need latency under llama.cpp you need something in the request path; see the
README's Python-or-Rust section.

**403 from a browser on another machine.** Two separate guards, both deliberate. Bind wider
*and* name the host you use:

```bash
FLEET_METRICS_BIND=0.0.0.0 FLEET_METRICS_ALLOWED_HOSTS=box.lan python3 fleet-metrics.py
```

`/metrics` is unauthenticated and reports GPU tenants, ARP neighbours, and established
connections. Put your own auth or a tunnel in front of it before exposing it to anything you
do not control.

## 7. Uninstall

```bash
systemctl --user disable --now llm-dashboard     # if you made the service
rm ~/.config/systemd/user/llm-dashboard.service
systemctl --user daemon-reload
```

Then delete the directory. Nothing is installed outside it — no packages, no dotfiles, no
system state, no registry entries. It writes nothing to disk while running.

---

*Verified on Python 3.12 against a 4×RTX 3090 box: default port, custom port, port collision,
invalid port values, and a deliberately broken `nvidia-smi` (page still serves, GPU list
empty, clear message on stderr).*
