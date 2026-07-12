#!/usr/bin/env python3
"""
fleet-metrics.py — lightweight metrics endpoint for the LLM serving dashboard.
Serves JSON from nvidia-smi + llama.cpp/vLLM /metrics + system stats.
Runs on :8092. No deps beyond the Python stdlib + nvidia-smi.
"""
import json
import os
import re
import subprocess
import time
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8092
LLAMA_METRICS_URL = "http://localhost:8001/metrics"
LLAMA_PROPS_URL = "http://localhost:8001/props"
COMFYUI_URL = "http://localhost:8188/"

# The primary worker isn't always on :8001 — a bench or a swap can land it on another port.
# Discover candidates DYNAMICALLY from listening sockets each resolution: :8001 first, then any
# listener in the serving range, probed for llama.cpp /props OR vLLM /v1/models. Excludes the
# 909x/809x bands and known non-server sidecars (a small CPU autocomplete on :8081 was otherwise
# picked as the primary while the real server sat on another port).
# Override the candidate list with WORKER_PORT_CANDIDATES=8001,8010 in the environment.
WORKER_PORT_CANDIDATES = [int(p) for p in os.environ.get(
    "WORKER_PORT_CANDIDATES", "8001,8010,8123").split(",") if p.strip()]
WORKER_EXCLUDE_PORTS = {
    8081,  # small CPU autocomplete sidecar — never the primary server
    8188,  # ComfyUI (image gen)
}


def _listening_ports():
    try:
        out = subprocess.check_output(["ss", "-ltn"], text=True, timeout=3)
        ports = set(int(m.group(1)) for m in re.finditer(r":(\d{4,5})\s", out))
        return sorted(p for p in ports if 8000 <= p <= 8299
                      and p not in WORKER_EXCLUDE_PORTS
                      and not 8090 <= p <= 8099)
    except Exception:
        return []


def _fetch_server_props(port):
    """Fetch server props from a port, trying llama.cpp /props first, then vLLM /v1/models."""
    props = fetch_llama_props(port)
    if not (isinstance(props, dict) and "_error" not in props):
        props = fetch_vllm_props(port)
    return props


def _port_nctx(port):
    """Return n_ctx for a valid server on *port*, or None if unreachable/wrong format."""
    props = _fetch_server_props(port)
    if isinstance(props, dict) and "_error" not in props:
        return props.get("n_ctx", 0) or 0
    return None


def resolve_worker_port():
    """The :8001 default wins when up; otherwise the responder with the LARGEST per-slot ctx.
    A large-context server usually wins, so a tiny-ctx utility server that slips past the
    exclusion list still loses to the real primary server."""
    seen = _listening_ports()
    candidates = [8001] + [p for p in (seen or WORKER_PORT_CANDIDATES) if p != 8001]
    best = None  # (n_ctx, port)
    for p in candidates:
        if p == 8001 and _port_nctx(p) is not None:
            return p  # :8001 wins immediately when alive
        nctx = _port_nctx(p)
        if nctx is not None and (best is None or nctx > best[0]):
            best = (nctx, p)
    return best[1] if best else 8001  # nothing up → report the default port as down

# Secondary CPU-only llama-servers (optional). Each raw llama-server exposes the prometheus
# /metrics, /props, /lora-adapters API on its own port; the dashboard shows one panel per entry.
# EDIT THESE for your setup, or set SECONDARY_SERVERS='name:port,name:port' in the environment.
# Leave empty to hide the panel. (Example defaults below assume nothing — they simply show 'down'
# until a server answers on that port.)
def _load_secondaries():
    env = os.environ.get("SECONDARY_SERVERS")
    if env is not None:
        out = []
        for item in env.split(","):
            if ":" in item:
                n, p = item.rsplit(":", 1)
                out.append({"name": n.strip(), "port": int(p), "model": ""})
        return out
    return [
        {"name": "cpu-worker-1", "port": 9093, "model": "example CPU llama-server"},
        {"name": "cpu-worker-2", "port": 9095, "model": "example CPU llama-server"},
    ]
DREAMERS = _load_secondaries()

def parse_nvidia_smi():
    """Query all GPUs and return structured data."""
    fields = [
        "index", "name", "utilization.gpu", "memory.used", "memory.total",
        "power.draw", "power.limit", "temperature.gpu", "fan.speed",
        "clocks.sm", "clocks.mem"
    ]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            text=True, timeout=5
        ).strip()
    except Exception as e:
        return [], str(e)

    gpus = []
    for line in out.split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 11:
            continue
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "gpu_util": float(parts[2]) if parts[2] != "[N/A]" else 0,
            "mem_used_mib": int(parts[3]) if parts[3] != "[N/A]" else 0,
            "mem_total_mib": int(parts[4]) if parts[4] != "[N/A]" else 0,
            "power_w": float(parts[5]) if parts[5] != "[N/A]" else 0,
            "power_limit_w": float(parts[6]) if parts[6] != "[N/A]" else 0,
            "temp_c": int(parts[7]) if parts[7] != "[N/A]" else 0,
            "fan_pct": int(parts[8]) if parts[8] != "[N/A]" else 0,
            "sm_clock_mhz": int(parts[9]) if parts[9] != "[N/A]" else 0,
            "mem_clock_mhz": int(parts[10]) if parts[10] != "[N/A]" else 0,
        })
    return gpus


def _safe_int(val, default=0):
    """Parse an integer from a string, returning *default* on failure."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _resolve_process_name(pid, raw_name):
    """Return a meaningful process name. If *raw_name* starts with 'python',
    read /proc/PID/cmdline to find the actual .py script."""
    name = os.path.basename(raw_name)
    if name.startswith("python"):
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().decode().split("\0")
            name = next((os.path.basename(a) for a in argv if a.endswith(".py")), name)
        except Exception:
            pass
    return name

def _query_gpu_compute_apps():
    """Query nvidia-smi for GPU UUIDs and compute apps.

    Returns (uuid_to_idx_dict, app_output_string) or None on any error.
    """
    try:
        uuid_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True, timeout=5).strip()
        uuid_to_idx = {}
        for line in uuid_out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                uuid_to_idx[parts[1]] = int(parts[0])
        app_out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
        return uuid_to_idx, app_out
    except Exception:
        return None


def attach_gpu_procs(gpus):
    """Annotate each GPU dict with its actual compute tenants (name + VRAM), so the
    dashboard labels cards from ground truth instead of VRAM-threshold guessing."""
    if not isinstance(gpus, list):  # parse_nvidia_smi error path returns ([], err)
        return gpus
    data = _query_gpu_compute_apps()
    if data is None:
        return gpus
    uuid_to_idx, app_out = data
    procs_by_idx = {}
    for line in app_out.split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or parts[0] not in uuid_to_idx:
            continue
        idx = uuid_to_idx[parts[0]]
        name = _resolve_process_name(parts[1], parts[2])
        procs_by_idx.setdefault(idx, []).append({
            "name": name, "pid": int(parts[1]), "mem_mib": _safe_int(parts[3])
        })
    for g in gpus:
        plist = procs_by_idx.get(g["index"], [])
        g["procs"] = plist
        g["tenant"] = " + ".join(sorted({p["name"] for p in plist})) if plist else ""
    return gpus


def parse_prometheus(text):
    """Parse Prometheus-format metrics text into a dict."""
    metrics = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # match "metric_name value" or "metric_name{labels} value"
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([\d.eE+-]+)", line)
        if m:
            name = m.group(1)
            val = float(m.group(2))
            metrics[name] = val
    return metrics


def fetch_llama_metrics(port=8001):
    """Fetch /metrics (Prometheus) from a llama-server."""
    try:
        req = urllib.request.Request(f"http://localhost:{port}/metrics", headers={"User-Agent": "curl"})
        resp = urllib.request.urlopen(req, timeout=10)
        text = resp.read().decode("utf-8", errors="replace")
        return parse_prometheus(text)
    except Exception as e:
        return {"_error": str(e)}


def fetch_llama_props(port=8001):
    """Fetch /props from a llama-server."""
    try:
        req = urllib.request.Request(f"http://localhost:{port}/props", headers={"User-Agent": "curl"})
        resp = urllib.request.urlopen(req, timeout=3)
        data = json.loads(resp.read())
        params = data.get("default_generation_settings", {}).get("params", {})
        return {
            "alias": data.get("model_alias", ""),
            "model_path": data.get("model_path", ""),
            "n_ctx": data.get("default_generation_settings", {}).get("n_ctx", 0),
            "total_slots": data.get("total_slots", 0),
            "temperature": params.get("temperature", 0),
            "top_p": params.get("top_p", 0),
            "top_k": params.get("top_k", 0),
            "min_p": params.get("min_p", 0),
        }
    except Exception as e:
        return {"_error": str(e)}


def fetch_llama_loras(port=8001):
    """Fetch /lora-adapters from a llama-server. Returns a list of {id,path,scale}."""
    try:
        req = urllib.request.Request(f"http://localhost:{port}/lora-adapters", headers={"User-Agent": "curl"})
        resp = urllib.request.urlopen(req, timeout=3)
        data = json.loads(resp.read())
        loras = []
        for a in (data if isinstance(data, list) else []):
            path = a.get("path", "")
            loras.append({
                "id": a.get("id"),
                "name": os.path.basename(path).replace(".gguf", "") if path else str(a.get("id")),
                "scale": a.get("scale", 0),
            })
        return loras
    except Exception:
        # endpoint absent or no adapters compiled in — treat as "none applied"
        return []


def fetch_vllm_props(port):
    """vLLM has no /props; synthesize the same shape from /v1/models."""
    try:
        req = urllib.request.Request(f"http://localhost:{port}/v1/models", headers={"User-Agent": "curl"})
        resp = urllib.request.urlopen(req, timeout=3)
        data = json.loads(resp.read())
        m = (data.get("data") or [{}])[0]
        return {
            "alias": m.get("id", ""),
            "model_path": m.get("root", ""),
            "n_ctx": m.get("max_model_len", 0),
            "total_slots": 0,
            "temperature": 0, "top_p": 0, "top_k": 0, "min_p": 0,
            "engine": "vllm",
        }
    except Exception as e:
        return {"_error": str(e)}


# Per-port counter history for vLLM rate derivation: {port: [(t, prompt_total, gen_total), ...]}
_VLLM_HIST = {}
_DECODE_WINDOW_S = 3.0    # short window so brief decode bursts read true (was 6.0; a 300-tok MTP
                          # generation is a ~4-5s burst that 6s smearing under-read to ~23 t/s)
# Prompt throughput = BURST-HOLD. Prefill happens in ~1s bursts; we compute
# the TRUE rate of the latest burst from vLLM's cumulative pair Δprefill_tokens/Δprefill_time
# and HOLD it between bursts. Seeded with the lifetime ratio so it's never 0 after first prefill.
# {port: (last_pf_tok, last_pf_time, held_rate)}
_VLLM_PP = {}
# spec-decode acceptance burst-hold: {port: (last_draft_total, held_pct)}
_VLLM_SPEC = {}


def _parse_vllm_metrics(metrics):
    """Extract vLLM metric values from a parsed Prometheus dict.

    Returns a dict with keys: kv_ratio, prompt_total, gen_total, running,
    pf_tok, pf_time, spec_acc, spec_draft.
    """
    kv_ratio = prompt_total = gen_total = running = 0.0
    pf_tok = pf_time = 0.0
    spec_acc = spec_draft = None   # spec-decode counters absent unless the server runs MTP/draft
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if k.startswith("vllm:kv_cache_usage_perc") or k.startswith("vllm:gpu_cache_usage_perc"):
                kv_ratio = v
            elif k.startswith("vllm:prompt_tokens_total"):
                prompt_total += v
            elif k.startswith("vllm:generation_tokens_total"):
                gen_total += v
            elif k.startswith("vllm:num_requests_running"):
                running += v
            elif k.startswith("vllm:request_prefill_kv_computed_tokens_sum"):
                pf_tok += v
            elif k.startswith("vllm:request_prefill_time_seconds_sum"):
                pf_time += v
            elif k.startswith("vllm:spec_decode_num_accepted_tokens_total"):
                spec_acc = (spec_acc or 0.0) + v
            elif k.startswith("vllm:spec_decode_num_draft_tokens_total"):
                spec_draft = (spec_draft or 0.0) + v
    return {
        "kv_ratio": kv_ratio, "prompt_total": prompt_total, "gen_total": gen_total,
        "running": running, "pf_tok": pf_tok, "pf_time": pf_time,
        "spec_acc": spec_acc, "spec_draft": spec_draft,
    }


def _update_vllm_history(port, prompt_total, gen_total, spec_acc, spec_draft):
    """Update vLLM history for a port, handling restarts and memory bounds.

    Returns the updated history list.
    """
    now = time.time()
    hist = _VLLM_HIST.setdefault(port, [])
    # server restart resets counters -> drop stale history so rates don't go negative-then-zero
    if hist and (prompt_total < hist[-1][1] or gen_total < hist[-1][2]):
        hist.clear()
    hist.append((now, prompt_total, gen_total, spec_acc or 0.0, spec_draft or 0.0))
    del hist[: max(0, len(hist) - 120)]   # bound memory (~4 min at 2s polls)
    return hist, now


def _compute_windowed_rate(hist, now, prompt_total, gen_total, idx, window):
    """Compute rate from history over a time window.

    idx=1 for prompt tokens, idx=2 for generation tokens.
    Returns rate as tokens/second (floored at 0).
    """
    ref = None
    for s in hist:
        if now - s[0] <= window:
            break
        ref = s
    ref = ref or hist[0]
    dt = now - ref[0]
    if dt <= 0:
        return 0.0
    cur = (prompt_total, gen_total)[idx - 1]
    return max(0.0, (cur - ref[idx]) / dt)


def _compute_prompt_tps(port, pf_tok, pf_time):
    """Compute prompt tokens/second using burst-hold from prefill-time histogram pair.

    Prefill happens in ~1s bursts; we compute the TRUE rate of the latest burst
    from Δprefill_tokens/Δprefill_time and HOLD it between bursts.
    Seeded with the lifetime ratio so it's never 0 after first prefill.
    """
    lt, lp, held = _VLLM_PP.get(port, (None, None, 0.0))
    if lt is not None and (pf_tok < lt or pf_time < lp):
        lt = lp = None                                     # server restart -> counter reset
    if lt is None:
        held = (pf_tok / pf_time) if pf_time > 0 else 0.0  # seed: lifetime average
    elif pf_tok > lt and pf_time > lp:
        held = (pf_tok - lt) / (pf_time - lp)              # rate of the latest burst
    _VLLM_PP[port] = (pf_tok, pf_time, held)
    return held


def _compute_spec_accept_pct(port, hist, now, spec_acc, spec_draft):
    """Compute MTP/spec-decode acceptance % using burst-hold.

    Decode bursts are short; 0 between generations is not a valid answer.
    Rate of the latest burst = Δaccepted/Δdrafted over the recent window;
    seeded with the lifetime ratio. None when the server runs no drafter.
    """
    spec_accept_pct = None
    if spec_draft is not None and spec_draft > 0:
        ref = None
        for s in hist:
            if now - s[0] <= 30.0:   # look back up to 30s for the latest burst
                break
            ref = s
        ref = ref or hist[0]
        d_acc, d_draft = (spec_acc or 0.0) - ref[3], spec_draft - ref[4]
        held_pct = _VLLM_SPEC.get(port, (None, None))[1]
        if d_draft > 0:
            held_pct = 100.0 * max(0.0, d_acc) / d_draft
        elif held_pct is None:
            held_pct = 100.0 * (spec_acc or 0.0) / spec_draft   # seed: lifetime average
        _VLLM_SPEC[port] = (spec_draft, held_pct)
        spec_accept_pct = round(held_pct, 1)
    return spec_accept_pct


def fetch_vllm_endpoint(port):
    """vLLM flavor of fetch_endpoint: same output shape, vllm: metric names.

    vLLM exports cumulative counters (prompt_tokens_total / generation_tokens_total),
    so PP/TG tok-s are derived from the delta since the previous poll (first
    poll after a restart reads 0 — settles by the second sample).
    """
    metrics = fetch_llama_metrics(port)  # generic Prometheus parse works for vllm: names too
    props = fetch_vllm_props(port)
    up = "_error" not in props
    n_ctx = props.get("n_ctx", 0) if up else 0

    # Parse vLLM metrics into named values
    parsed = _parse_vllm_metrics(metrics)
    kv_ratio = parsed["kv_ratio"]
    prompt_total = parsed["prompt_total"]
    gen_total = parsed["gen_total"]
    running = parsed["running"]
    pf_tok = parsed["pf_tok"]
    pf_time = parsed["pf_time"]
    spec_acc = parsed["spec_acc"]
    spec_draft = parsed["spec_draft"]

    # Update history (handles restart detection, memory bounds)
    hist, now = _update_vllm_history(port, prompt_total, gen_total, spec_acc, spec_draft)

    # Decode TPS from windowed rate
    decode_tps = _compute_windowed_rate(hist, now, prompt_total, gen_total, 2, _DECODE_WINDOW_S)

    # Prompt TPS from burst-hold
    prompt_tps = _compute_prompt_tps(port, pf_tok, pf_time)

    # Spec-decode acceptance %
    spec_accept_pct = _compute_spec_accept_pct(port, hist, now, spec_acc, spec_draft)

    return {
        "status": "up" if up else "down",
        "metrics": metrics,
        "props": props,
        "loras": [],
        "derived": {
            "prompt_tps": round(prompt_tps, 1),
            "decode_tps": round(decode_tps, 1),
            "spec_accept_pct": spec_accept_pct,
            "n_ctx": n_ctx,
            "ctx_used_tokens": round(kv_ratio * n_ctx),
            "ctx_fill_pct": round(kv_ratio * 100, 1),
            "n_loras_active": 0,
            "n_loras": 0,
            "requests_processing": running,
        },
    }


def fetch_endpoint(port):
    """Full snapshot of one llama-server: status + metrics + props + loras.

    Derives the convenience fields the dashboard graphs: PP/TG tok-s,
    context length, context %fill (KV usage), and the active LoRA count.
    A port without llama.cpp /props but with /v1/models is a vLLM server
    (e.g. a large model on an alternate port) → routed to fetch_vllm_endpoint.
    """
    metrics = fetch_llama_metrics(port)
    props = fetch_llama_props(port)
    if isinstance(props, dict) and "_error" in props:
        if "_error" not in fetch_vllm_props(port):
            return fetch_vllm_endpoint(port)
    up = ("_error" not in metrics) or ("_error" not in props)
    loras = fetch_llama_loras(port) if up else []
    active_loras = [l for l in loras if (l.get("scale") or 0) > 0]
    n_ctx = props.get("n_ctx", 0) if isinstance(props, dict) else 0
    kv_ratio = metrics.get("llamacpp:kv_cache_usage_ratio", 0) if isinstance(metrics, dict) else 0
    return {
        "status": "up" if up else "down",
        "metrics": metrics,
        "props": props,
        "loras": loras,
        "derived": {
            "prompt_tps": metrics.get("llamacpp:prompt_tokens_seconds", 0) if up else 0,
            "decode_tps": metrics.get("llamacpp:predicted_tokens_seconds", 0) if up else 0,
            "n_ctx": n_ctx,
            "ctx_used_tokens": metrics.get("llamacpp:kv_cache_tokens", round(kv_ratio * n_ctx)) if up else 0,
            "ctx_fill_pct": round(kv_ratio * 100, 1) if up else 0,
            "n_loras_active": len(active_loras),
            "n_loras": len(loras),
            "requests_processing": metrics.get("llamacpp:requests_processing", 0) if up else 0,
        },
    }


def check_port(host, port):
    """Check if a TCP port is accepting connections."""
    try:
        req = urllib.request.Request(f"http://localhost:{port}/", headers={"User-Agent": "curl"})
        resp = urllib.request.urlopen(req, timeout=2)
        return {"status": "up", "code": resp.status}
    except urllib.error.HTTPError as e:
        # 404/403 still means the server is up
        return {"status": "up", "code": e.code}
    except Exception:
        return {"status": "down", "code": None}


# Network rate state: {iface: (t, rx_bytes, tx_bytes)} from the previous poll —
# rates are derived per-request from the /proc/net/dev counter deltas.
_NET_LAST = {}
_NET_SKIP = ("lo", "docker", "veth", "br-", "virbr", "tailscale", "tun", "wg")


def get_network():
    """Per-physical-NIC RX/TX rates in bytes/s (delta since last poll) + lifetime totals."""
    now = time.time()
    ifaces = []
    try:
        with open("/proc/net/dev") as f:
            lines = f.readlines()[2:]
    except Exception:
        return {"ifaces": [], "rx_bps": 0, "tx_bps": 0}
    total_rx_bps = total_tx_bps = 0.0
    for line in lines:
        name, _, rest = line.partition(":")
        name = name.strip()
        if not rest or any(name.startswith(s) for s in _NET_SKIP):
            continue
        cols = rest.split()
        rx, tx = int(cols[0]), int(cols[8])
        last = _NET_LAST.get(name)
        rx_bps = tx_bps = 0.0
        if last and now > last[0] and rx >= last[1] and tx >= last[2]:
            dt = now - last[0]
            rx_bps = (rx - last[1]) / dt
            tx_bps = (tx - last[2]) / dt
        _NET_LAST[name] = (now, rx, tx)
        total_rx_bps += rx_bps
        total_tx_bps += tx_bps
        ifaces.append({"name": name, "rx_bps": round(rx_bps), "tx_bps": round(tx_bps),
                       "rx_total": rx, "tx_total": tx})
    # busiest first so the dashboard can show the active NIC's name
    ifaces.sort(key=lambda i: i["rx_bps"] + i["tx_bps"], reverse=True)
    return {"ifaces": ifaces, "rx_bps": round(total_rx_bps), "tx_bps": round(total_tx_bps)}


# LAN metadata — PASSIVE only (no probing/scanning): the kernel ARP table (`ip neigh`,
# devices this box has actually exchanged packets with) + established TCP connections
# (`ss -tn`). Names resolved from /etc/hosts, never live DNS. Cached 10s so the 2s
# dashboard poll stays cheap.
_LAN_CACHE = {"t": 0.0, "data": None}
_PRIVATE_PREFIXES = ("10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.",
                     "172.2", "172.30.", "172.31.")


def _hosts_names():
    names = {}
    try:
        for line in open("/etc/hosts"):
            parts = line.split("#")[0].split()
            if len(parts) >= 2 and not parts[0].startswith("127."):
                names[parts[0]] = parts[1]
    except Exception:
        pass
    return names


def _parse_arp_neighbors(names):
    """Parse `ip -4 neigh show` output into a list of neighbor dicts.

    Filters out skipped interfaces and FAILED entries. Sorted by REACHABLE first.
    """
    neighbors = []
    try:
        out = subprocess.check_output(["ip", "-4", "neigh", "show"], text=True, timeout=3)
        for line in out.strip().split("\n"):
            parts = line.split()
            if len(parts) < 2 or "dev" not in parts:
                continue
            dev = parts[parts.index("dev") + 1]
            if any(dev.startswith(s) for s in _NET_SKIP):
                continue
            mac = parts[parts.index("lladdr") + 1] if "lladdr" in parts else ""
            state = parts[-1]
            if state == "FAILED" or not mac:
                continue
            neighbors.append({"ip": parts[0], "name": names.get(parts[0], ""),
                              "mac": mac, "state": state, "dev": dev})
    except Exception:
        pass
    neighbors.sort(key=lambda n: (n["state"] != "REACHABLE", n["ip"]))
    return neighbors


def _service_ports(lport, pport):
    """Return service-side ports (low/registered, < 30000) from local and peer ports."""
    return {int(prt) for prt in (lport, pport) if prt.isdigit() and int(prt) < 30000}


def _update_lan_peer(ip, names, lan_peers, lport, pport):
    """Add or update a LAN peer entry in the lan_peers dict."""
    e = lan_peers.setdefault(ip, {"ip": ip, "name": names.get(ip, ""),
                                   "conns": 0, "ports": set()})
    e["conns"] += 1
    e["ports"].update(_service_ports(lport, pport))


def _parse_tcp_peers(names):
    """Parse `ss -Htn state established` into LAN peers and WAN connection count.

    Returns (peers_list, wan_conns_count). Peers are sorted by connection count desc,
    with up to 6 service ports each.
    """
    lan_peers, wan_conns = {}, 0
    try:
        out = subprocess.check_output(["ss", "-Htn", "state", "established"],
                                      text=True, timeout=3)
        for line in out.strip().split("\n"):
            cols = line.split()
            if len(cols) < 4:
                continue
            local, peer = cols[2], cols[3]
            pip, _, pport = peer.rpartition(":")
            lport = local.rpartition(":")[2]
            if pip.startswith("127.") or pip.startswith("[") or pip == "":
                continue
            if any(pip.startswith(p) for p in _PRIVATE_PREFIXES):
                _update_lan_peer(pip, names, lan_peers, lport, pport)
            else:
                wan_conns += 1
    except Exception:
        pass
    peers = sorted(lan_peers.values(), key=lambda p: -p["conns"])
    for p in peers:
        p["ports"] = sorted(p["ports"])[:6]
    return peers, wan_conns


def get_lan():
    """Gather LAN metadata: ARP neighbors + established TCP connections.

    Cached for 10 seconds so the 2-second dashboard poll stays cheap.
    """
    now = time.time()
    if _LAN_CACHE["data"] is not None and now - _LAN_CACHE["t"] < 10:
        return _LAN_CACHE["data"]
    names = _hosts_names()
    neighbors = _parse_arp_neighbors(names)
    peers, wan_conns = _parse_tcp_peers(names)
    data = {"neighbors": neighbors, "lan_peers": peers, "wan_conns": wan_conns,
            "reachable": sum(1 for n in neighbors if n["state"] == "REACHABLE"),
            "total_devices": len(neighbors)}
    _LAN_CACHE.update(t=now, data=data)
    return data


def get_system():
    """RAM + loadavg."""
    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    meminfo[parts[0]] = int(parts[1].strip().split()[0])  # kB
        total = meminfo.get("MemTotal", 0)
        avail = meminfo.get("MemAvailable", 0)
        with open("/proc/loadavg") as f:
            load = f.read().strip().split()
        return {
            "ram_total_mb": total // 1024,
            "ram_used_mb": (total - avail) // 1024,
            "ram_avail_mb": avail // 1024,
            "load_1min": float(load[0]),
            "load_5min": float(load[1]),
            "load_15min": float(load[2]),
        }
    except Exception:
        return {}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics" or self.path == "/api/metrics":
            self._handle_metrics()
        elif self.path == "/models":
            self._handle_models()
        elif self.path == "/health":
            self._handle_health()
        else:
            self._handle_not_found()

    def _send_json(self, code, body):
        """Send a JSON response with CORS headers."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_metrics(self):
        payload = self._gather()
        body = json.dumps(payload, indent=2).encode()
        self._send_json(200, body)

    def _handle_models(self):
        # loadable-model registry for the dashboard MODEL LIBRARY (edit models-registry.json)
        try:
            reg = os.environ.get("MODELS_REGISTRY",
                                 os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "models-registry.json"))
            body = open(os.path.expanduser(reg), "rb").read()
            self._send_json(200, body)
        except Exception as e:
            body = json.dumps({"_error": str(e)}).encode()
            self._send_json(500, body)

    def _handle_health(self):
        body = b'{"status":"ok"}'
        self._send_json(200, body)

    def _handle_not_found(self):
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"404 - use /metrics")

    def _read_thoughts(self, max_bytes=6000):
        """Tail of an optional Thought-Tap log — live reasoning_content (CoT) captured by a
        proxy in front of the worker. Set THOUGHT_LOG=/path/to/thinking.log to enable it;
        empty (feature off) when unset or the file is absent."""
        path = os.path.expanduser(os.environ.get("THOUGHT_LOG", ""))
        if not path:
            return {"text": "", "mtime": 0, "bytes": 0}
        try:
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                if size > max_bytes:
                    f.seek(size - max_bytes)
                    f.readline()  # drop the partial first line
                text = f.read().decode("utf-8", "replace")
            return {"text": text, "mtime": os.path.getmtime(path), "bytes": size}
        except FileNotFoundError:
            return {"text": "", "mtime": 0, "bytes": 0}
        except Exception as e:
            return {"text": f"(thought tap read error: {e})", "mtime": 0, "bytes": 0}

    def _fetch_thought_streams(self, port=8090):
        """Structured per-request CoT streams from an optional reasoning-tap proxy (/thoughts).
        Each concurrent thinker is its own stream → its own dashboard panel. [] if tap down."""
        try:
            req = urllib.request.Request(f"http://localhost:{port}/thoughts", headers={"User-Agent": "curl"})
            with urllib.request.urlopen(req, timeout=1.0) as r:
                return json.loads(r.read().decode("utf-8", "replace")).get("streams", [])
        except Exception:
            return []

    def _gather_dreamers(self):
        """Fetch snapshots for each secondary (dreamer) server."""
        dreamers = []
        for d in DREAMERS:
            snap = fetch_endpoint(d["port"])
            snap["name"] = d["name"]
            snap["port"] = d["port"]
            snap["model"] = d["model"]
            dreamers.append(snap)
        return dreamers

    def _gather(self):
        # Primary worker — keep the llama_8001 key shape, now also carrying
        # loras + derived fields so the worker panel can show ctx-fill / LoRAs too.
        worker_port = resolve_worker_port()
        worker = fetch_endpoint(worker_port)
        return {
            "timestamp": time.time(),
            "gpus": attach_gpu_procs(parse_nvidia_smi()),
            "worker_port": worker_port,
            "llama_8001": worker,
            "dreamers": self._gather_dreamers(),
            "services": {
                "llama_8001": check_port("localhost", worker_port),
                "comfyui_8188": check_port("localhost", 8188),
            },
            "system": get_system(),
            "network": get_network(),
            "lan": get_lan(),
            "thoughts": self._read_thoughts(),
            "thought_streams": self._fetch_thought_streams(),
        }

    def log_message(self, format, *args):
        pass  # suppress request logging


if __name__ == "__main__":
    print(f"fleet-metrics serving on :{PORT}/metrics")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
