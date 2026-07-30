#!/usr/bin/env python3
"""
fleet-metrics.py — lightweight metrics endpoint for the LLM serving dashboard.
Serves JSON from nvidia-smi + llama.cpp/vLLM /metrics + system stats.
Runs on :8092. No deps beyond the Python stdlib + nvidia-smi.
"""
import ipaddress
import json
import math
import os
import socket
import re
import subprocess
import sys
import time
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8092

# Loopback by default. /metrics exposes GPU tenants, system stats, ARP neighbours and established
# connections — that is not something to put on a LAN implicitly. Opt in explicitly if you want it
# reachable from another machine, and put it behind something that authenticates if you do:
#   FLEET_METRICS_BIND=0.0.0.0 python3 fleet-metrics.py
BIND = os.environ.get("FLEET_METRICS_BIND", "127.0.0.1")

# Serve a canned payload instead of probing the machine. Used to regenerate docs/screenshot.png
# from docs/example-metrics.json, so the README's "rendered with example data" caption is
# reproducible by anyone rather than something you have to take on trust — and so the hero image
# can never leak a real model name, MAC address or LAN topology:
#   FLEET_METRICS_FIXTURE=docs/example-metrics.json python3 fleet-metrics.py
FIXTURE = os.environ.get("FLEET_METRICS_FIXTURE", "")

# This server also serves index.html (GET /), so the dashboard is SAME-ORIGIN and needs no CORS
# at all. That is the whole point: `Access-Control-Allow-Origin: *` lets any site you visit read
# your telemetry, and `null` is no better — every sandboxed iframe gets Origin: null, so allowing
# it re-opens the same hole to any page that can embed one. Binding to loopback does not help
# either; the request comes from your own browser.
#
# Opening index.html from file:// therefore no longer works by default, and that is deliberate.
# If you must, `null` can be named explicitly in FLEET_METRICS_ALLOWED_ORIGINS — understand that
# it grants read access to any opaque-origin document, including a hostile sandboxed iframe.
_ALLOWED_ORIGINS = {o.strip() for o in os.environ.get(
    "FLEET_METRICS_ALLOWED_ORIGINS", "").split(",") if o.strip()}
_LOCAL_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")


def cors_origin(origin):
    """Echo `origin` only if allowlisted; otherwise None (send no CORS header at all)."""
    if not origin:
        return None
    if _LOCAL_ORIGIN.match(origin) or origin in _ALLOWED_ORIGINS:
        return origin
    return None


# --- DNS-rebinding guard -------------------------------------------------------------------
# CORS cannot stop this attack, which is why it needs its own defence. A page on
# http://rebind.example:8092 whose DNS flips to 127.0.0.1 becomes SAME-ORIGIN with this server —
# no CORS check is even consulted, and loopback binding does not help because the request comes
# from the victim's own browser. The one thing that still distinguishes it is the Host header,
# which carries the attacker's name rather than a loopback name. So: only serve requests whose
# Host we expect.
_ALLOWED_HOSTS = {h.strip().lower() for h in os.environ.get(
    "FLEET_METRICS_ALLOWED_HOSTS", "").split(",") if h.strip()}
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _host_only(host_header):
    """Host header -> bare host, port stripped, IPv6 brackets removed.

    Returns a BRACKETLESS form for both families so comparisons cannot disagree about brackets
    (`[2001:db8::10]` vs `2001:db8::10` previously compared unequal, rejecting a deliberate
    IPv6 bind).
    """
    h = host_header.strip()
    if h.startswith("["):                       # [::1]:8092 -> ::1
        end = h.find("]")
        return (h[1:end] if end != -1 else h.strip("[]")).lower()
    # A bare IPv6 literal contains many colons; only strip a port from host:port.
    if h.count(":") > 1:
        return h.lower()
    return (h.rsplit(":", 1)[0] if ":" in h else h).lower()


def host_allowed(host_header):
    """Allow loopback names, any IP literal, and explicitly configured names.

    DNS rebinding needs a NAME whose resolution can be flipped. An IP literal cannot be rebound:
    for the attacker's page to be same-origin with `http://<ip>:8092` it must already be served
    from that address. So literals are safe to accept, and accepting them is what makes the
    documented LAN workflow work — binding 0.0.0.0 and browsing to the machine's own address used
    to return 403, i.e. the README described something the guard forbade.

    Names still have to be loopback or named in FLEET_METRICS_ALLOWED_HOSTS.
    """
    if not host_header:
        return False        # HTTP/1.1 requires Host; absence is not a browser we need to serve
    h = _host_only(host_header)
    if h in _LOOPBACK_HOSTS or h in _ALLOWED_HOSTS:
        return True
    try:
        ipaddress.ip_address(h)
        return True                              # an address literal; not rebindable
    except ValueError:
        return False                             # a name we were not told to expect


# --- scraper transport -------------------------------------------------------------------
# Every URL we scrape is a literal localhost + integer port, so an upstream cannot point us
# anywhere directly. A REDIRECT can: a listener answering /props with
# `302 Location: http://169.254.169.254/latest/meta-data/` would have this box fetch that on
# every poll. Refuse redirects outright rather than trying to validate hops (DNS rebinding,
# encoded IPs, IPv6 and userinfo all make hop validation a losing game).
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # -> urllib raises HTTPError; callers already treat that as "not usable"


# ProxyHandler({}) is NOT redundant. build_opener() installs a default ProxyHandler that reads
# HTTP_PROXY/http_proxy from the environment, so on a box with a corporate proxy configured these
# "localhost" scrapes would be handed to that proxy — the opposite of the local-only guarantee the
# README makes. An empty mapping pins every scrape to a direct connection.
_OPENER = urllib.request.build_opener(_NoRedirect, urllib.request.ProxyHandler({}))

# Bound the body. The server is single-threaded, so one upstream returning an endless stream
# would otherwise wedge /metrics and /health for every client, and a huge body would exhaust
# memory. read(N+1) lets us detect overflow without materialising more than the cap.
MAX_SCRAPE_BYTES = 8 * 1024 * 1024


def scrape_open(req, timeout):
    """urlopen for scraper targets: no redirects followed."""
    return _OPENER.open(req, timeout=timeout)


SCRAPE_DEADLINE_S = 10.0   # total wall-clock budget for reading one upstream body


def read_capped(resp, limit=MAX_SCRAPE_BYTES, deadline_s=SCRAPE_DEADLINE_S):
    """Read at most `limit` bytes within `deadline_s` wall-clock seconds, then close.

    The socket timeout passed to urlopen is an INACTIVITY timeout: a peer that dribbles one byte
    just inside it never trips it, and `resp.read(n)` would block indefinitely. This server is
    single-threaded, so that one upstream wedges /metrics, /health, /models and / for every
    client. Read in chunks against a monotonic deadline so total time is bounded regardless of
    how the bytes are paced, and always close the response so sockets are not leaked on the
    error paths.
    """
    chunks, total, end = [], 0, time.monotonic() + deadline_s
    try:
        while total <= limit:
            if time.monotonic() > end:
                raise TimeoutError(f"upstream exceeded {deadline_s}s read budget")
            want = min(65536, limit + 1 - total)
            # read1() performs ONE socket read and returns whatever arrived. resp.read(n) instead
            # blocks until it has all n bytes, so a trickling peer would sit inside a single call
            # for as long as it liked and the deadline above would never be re-checked — the
            # chunking would be decorative. Fall back to read() only if read1 is unavailable.
            reader = getattr(resp, "read1", None)
            chunk = reader(want) if reader is not None else resp.read(want)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > limit:
            raise ValueError(f"upstream response exceeded {limit} bytes")
        return b"".join(chunks)
    finally:
        try:
            resp.close()
        except Exception:
            pass
LLAMA_METRICS_URL = "http://localhost:8001/metrics"
LLAMA_PROPS_URL = "http://localhost:8001/props"
COMFYUI_URL = "http://localhost:8188/"

# The primary worker isn't always on :8001 — a bench or a swap can land it on another port.
# Discover candidates DYNAMICALLY from listening sockets each resolution: :8001 first, then any
# listener in the serving range, probed for llama.cpp /props OR vLLM /v1/models. Excludes the
# 909x/809x bands and known non-server sidecars (a small CPU autocomplete on :8081 was otherwise
# picked as the primary while the real server sat on another port).
# Override the candidate list with WORKER_PORT_CANDIDATES=8001,8010 in the environment.
_WORKER_PORTS_ENV = os.environ.get("WORKER_PORT_CANDIDATES")
WORKER_PORT_CANDIDATES = [int(p) for p in (_WORKER_PORTS_ENV or "8001,8010,8123").split(",")
                          if p.strip()]
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


def resolve_worker_port():
    """Pick the primary worker port.

    Default: :8001 wins outright when it answers; otherwise the responder with the LARGEST
    per-slot ctx, so a tiny-ctx utility server that slips past the exclusion list still loses to
    the real primary.

    If WORKER_PORT_CANDIDATES is set, that list is used EXCLUSIVELY and in order. It used to be
    consulted only when socket discovery happened to find nothing, so an explicit
    `WORKER_PORT_CANDIDATES=9000` was silently ignored whenever any unrelated listener existed in
    the scan range — configuration that looks applied and isn't.
    """
    if _WORKER_PORTS_ENV:
        candidates = list(dict.fromkeys(WORKER_PORT_CANDIDATES))   # de-dup, order preserved
    else:
        seen = _listening_ports()
        candidates = [8001] + [p for p in (seen or WORKER_PORT_CANDIDATES) if p != 8001]
    best = None  # (n_ctx, port)
    for p in candidates:
        props = fetch_llama_props(p)
        if not (isinstance(props, dict) and "_error" not in props):
            props = fetch_vllm_props(p)
        if isinstance(props, dict) and "_error" not in props:
            if p == 8001:
                return p
            n_ctx = props.get("n_ctx", 0) or 0
            if best is None or n_ctx > best[0]:
                best = (n_ctx, p)
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
SECONDARIES = _load_secondaries()

def _to_int(s, default=0):
    """Parse an nvidia-smi field to int, tolerating '[N/A]', '[Not Supported]',
    'ERR!', blanks, '75.0'-style floats, and non-finite junk ('inf'/'nan'/'1e309').
    Never raises."""
    try:
        v = float(s)
    except (ValueError, TypeError):
        return default
    if not math.isfinite(v):  # 'inf'/'nan' → int() would OverflowError / ValueError
        return default
    return int(v)


def _to_float(s, default=0.0):
    """As _to_int but returns a float. Non-finite values collapse to the default so
    they never reach json.dumps (which would emit Infinity/NaN — invalid JSON that
    breaks the browser's JSON.parse)."""
    try:
        v = float(s)
    except (ValueError, TypeError):
        return default
    return v if math.isfinite(v) else default


def _json_safe(obj):
    """Recursively replace non-finite floats (NaN/Infinity) with None. An upstream
    server's /props or /metrics can carry these — Python's json.loads accepts them,
    but json.dumps would then emit bare `NaN`/`Infinity`, which is invalid JSON and
    makes the browser's resp.json() throw. Sanitizing at the serialization sink means
    no single bad upstream value can break the whole dashboard."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# nvidia-smi's clocks_throttle_reasons.active bitmask, decoded to short labels. GpuIdle (0x1) is
# deliberately absent: it is a reason the clocks are low, not a reason the card is being held back,
# and flagging every idle card as "throttled" would train you to ignore the field entirely.
_THROTTLE_BITS = (
    (0x0000000000000002, "app_clocks"),
    (0x0000000000000004, "sw_power_cap"),
    (0x0000000000000008, "hw_slowdown"),
    (0x0000000000000010, "sync_boost"),
    (0x0000000000000020, "sw_thermal"),
    (0x0000000000000040, "hw_thermal"),
    (0x0000000000000080, "hw_power_brake"),
    (0x0000000000000100, "display_clocks"),
)

# Fields every supported driver has, and the extras added later (PCIe link state + throttle
# reasons). The extras are queried in the SAME call when supported and dropped wholesale when not:
# nvidia-smi fails the whole query on one unknown field name, so a driver that renamed
# clocks_throttle_reasons.* (newer builds prefer clocks_event_reasons.*) would otherwise take the
# entire GPU panel down with it rather than losing one line of detail.
_SMI_BASE_FIELDS = [
    "index", "name", "utilization.gpu", "memory.used", "memory.total",
    "power.draw", "power.limit", "temperature.gpu", "fan.speed",
    "clocks.sm", "clocks.mem",
]
_SMI_EXTRA_FIELDS = [
    "pcie.link.gen.current", "pcie.link.gen.max",
    "pcie.link.width.current", "pcie.link.width.max",
    "clocks_throttle_reasons.active",
]
_SMI_EXTRAS_OK = True   # flipped off for the process lifetime the first time the extras fail


def _query_nvidia_smi(fields):
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
        text=True, timeout=5
    ).strip()


def parse_nvidia_smi():
    """Query all GPUs and return a list of structured dicts.

    Always returns a list (empty on failure), never a tuple — so a downstream
    `for gpu of gpus` loop is always safe. Individual unparseable fields (e.g.
    nvidia-smi emitting '[Not Supported]' or 'ERR!' for power/clocks on some
    cards) degrade to 0 instead of 500-ing the whole /metrics response."""
    global _SMI_EXTRAS_OK
    fields = _SMI_BASE_FIELDS + (_SMI_EXTRA_FIELDS if _SMI_EXTRAS_OK else [])
    try:
        out = _query_nvidia_smi(fields)
    except Exception as e:
        if _SMI_EXTRAS_OK:
            # Retry once without the optional fields before giving up on the panel.
            _SMI_EXTRAS_OK = False
            print(f"[fleet-metrics] nvidia-smi extended query failed ({e}); "
                  "falling back to base fields (no PCIe link / throttle detail)", file=sys.stderr)
            fields = _SMI_BASE_FIELDS
            try:
                out = _query_nvidia_smi(fields)
            except Exception as e2:
                print(f"[fleet-metrics] nvidia-smi query failed: {e2}", file=sys.stderr)
                return []
        else:
            print(f"[fleet-metrics] nvidia-smi query failed: {e}", file=sys.stderr)
            return []

    gpus = []
    for line in out.split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(_SMI_BASE_FIELDS):
            continue
        gpu = {
            "index": _to_int(parts[0]),
            "name": parts[1],
            "gpu_util": _to_float(parts[2]),
            "mem_used_mib": _to_int(parts[3]),
            "mem_total_mib": _to_int(parts[4]),
            "power_w": _to_float(parts[5]),
            "power_limit_w": _to_float(parts[6]),
            "temp_c": _to_int(parts[7]),
            "fan_pct": _to_int(parts[8]),
            "sm_clock_mhz": _to_int(parts[9]),
            "mem_clock_mhz": _to_int(parts[10]),
        }
        if len(parts) >= len(_SMI_BASE_FIELDS) + len(_SMI_EXTRA_FIELDS):
            # Link state is what the card NEGOTIATED, not what the slot is wired for, and on a
            # multi-GPU board the two routinely differ (a card in an x1 slot reads x1 here while
            # advertising x16 max). That gap is the whole point of showing it: it explains a slow
            # tensor-parallel card that looks healthy on every other gauge.
            gpu["pcie_gen"] = _to_int(parts[11])
            gpu["pcie_gen_max"] = _to_int(parts[12])
            gpu["pcie_width"] = _to_int(parts[13])
            gpu["pcie_width_max"] = _to_int(parts[14])
            raw = parts[15]
            try:
                mask = int(raw, 16) if raw.lower().startswith("0x") else int(raw)
            except ValueError:
                mask = 0          # '[Not Supported]' / 'ERR!' — no reasons, not a fake zero clock
            gpu["throttle_mask"] = mask
            gpu["throttle"] = [name for bit, name in _THROTTLE_BITS if mask & bit]
        gpus.append(gpu)
    return gpus


def attach_gpu_procs(gpus):
    """Annotate each GPU dict with its actual compute tenants (name + VRAM), so the
    dashboard labels cards from ground truth instead of VRAM-threshold guessing."""
    if not isinstance(gpus, list):  # defensive: parse_nvidia_smi always returns a list
        return gpus
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
    except Exception:
        return gpus
    procs_by_idx = {}
    for line in app_out.split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or parts[0] not in uuid_to_idx:
            continue
        idx = uuid_to_idx[parts[0]]
        name = os.path.basename(parts[2])
        if name.startswith("python"):
            # a bare "python" tenant is useless — pull the script name from the cmdline
            try:
                argv = open(f"/proc/{parts[1]}/cmdline", "rb").read().decode().split("\0")
                name = next((os.path.basename(a) for a in argv if a.endswith(".py")), name)
            except Exception:
                pass
        try:
            mem = int(parts[3])
        except ValueError:
            mem = 0
        try:
            pid = int(parts[1])
        except ValueError:
            continue  # skip a record with a non-numeric PID rather than 500 the endpoint
        procs_by_idx.setdefault(idx, []).append({"name": name, "pid": pid, "mem_mib": mem})
    for g in gpus:
        plist = procs_by_idx.get(g["index"], [])
        g["procs"] = plist
        g["tenant"] = " + ".join(sorted({p["name"] for p in plist})) if plist else ""
    return gpus


# Counters and count-gauges add up across label sets; ratios, percentages and utilisation gauges
# do not. Anything not listed here is treated as non-additive and collapsed with max().
_ADDITIVE_SUFFIXES = ("_total", "_count", "_sum")
_ADDITIVE_NAMES = frozenset({
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "llamacpp:requests_processing",
    "llamacpp:requests_deferred",
})


def parse_prometheus(text):
    """Parse Prometheus-format metrics text into a dict.

    Series that differ only by labels are SUMMED into the bare metric name. They previously
    collapsed to whichever line appeared last, so a server exposing per-model counters (vLLM does)
    reported a single model's total as the whole total — and the consumer's `+=` had nothing left
    to add up. Keys stay bare names because the llama.cpp path looks them up exactly.

    Histogram buckets and quantiles are excluded from summing: adding across `le`/`quantile` is
    meaningless, so those keep the previous last-wins behaviour rather than inventing a number.
    """
    metrics = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # "metric_name value", "metric_name{labels} value", optional trailing timestamp.
        # The value is anchored to end-of-line: the old pattern was a prefix match, so a
        # corrupt sample like "93oops" was silently accepted as 93.
        m = re.match(
            r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?[ \t]+([\d.eE+-]+)(?:[ \t]+\d+)?[ \t]*$",
            line)
        if not m:
            continue
        name, labels = m.group(1), m.group(2) or ""
        try:
            val = float(m.group(3))
        except ValueError:
            continue  # a stray '+'/'-'/'.' token — skip this sample, keep the scrape
        if not math.isfinite(val):  # drop +Inf/NaN buckets rather than poison the JSON
            continue
        # Only ADDITIVE series may be summed. Summing everything non-bucket was a regression:
        # two `vllm:gpu_cache_usage_perc` series at 0.6 and 0.7 became 1.3, rendering as 130%
        # context fill and 1.3 x n_ctx used tokens — a confidently impossible number, which is
        # the exact failure this parser is supposed to prevent. Ratios and utilisation gauges
        # collapse to the maximum across label sets instead: bounded, and the honest "worst
        # card" reading when several models share an engine.
        additive = name.endswith(_ADDITIVE_SUFFIXES) or name in _ADDITIVE_NAMES
        if name not in metrics:
            metrics[name] = val
        elif additive:
            metrics[name] += val
        else:
            metrics[name] = max(metrics[name], val)
    return metrics


def fetch_llama_metrics(port=8001):
    """Fetch /metrics (Prometheus) from a llama-server."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", headers={"User-Agent": "curl"})
        resp = scrape_open(req, timeout=10)
        text = read_capped(resp).decode("utf-8", errors="replace")
        return parse_prometheus(text)
    except Exception as e:
        return {"_error": str(e)}


def fetch_llama_props(port=8001):
    """Fetch /props from a llama-server."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/props", headers={"User-Agent": "curl"})
        resp = scrape_open(req, timeout=3)
        data = json.loads(read_capped(resp))
        # Coerce every numeric field through _to_int/_to_float. These values are arithmetic
        # operands downstream (e.g. round(kv_ratio * n_ctx)); a server returning "n_ctx": "x"
        # would otherwise raise TypeError mid-scrape and blank the ENTIRE fleet, not just itself.
        # Shapes are guarded too — `default_generation_settings: []` would break .get().
        if not isinstance(data, dict):
            return {"_error": "props payload was not an object"}
        dgs = data.get("default_generation_settings")
        dgs = dgs if isinstance(dgs, dict) else {}
        params = dgs.get("params")
        params = params if isinstance(params, dict) else {}
        return {
            "alias": str(data.get("model_alias", "") or ""),
            "model_path": str(data.get("model_path", "") or ""),
            "n_ctx": _to_int(dgs.get("n_ctx", 0)),
            "total_slots": _to_int(data.get("total_slots", 0)),
            "temperature": _to_float(params.get("temperature", 0)),
            "top_p": _to_float(params.get("top_p", 0)),
            "top_k": _to_int(params.get("top_k", 0)),
            "min_p": _to_float(params.get("min_p", 0)),
        }
    except Exception as e:
        return {"_error": str(e)}


def fetch_llama_loras(port=8001):
    """Fetch /lora-adapters from a llama-server. Returns a list of {id,path,scale}."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/lora-adapters", headers={"User-Agent": "curl"})
        resp = scrape_open(req, timeout=3)
        data = json.loads(read_capped(resp))
        loras = []
        for a in (data if isinstance(data, list) else []):
            if not isinstance(a, dict):
                continue  # a bare string/number in the list would blow up .get() below
            path = str(a.get("path", "") or "")
            # scale is compared with `> 0` downstream — a string there raises TypeError.
            loras.append({
                "id": a.get("id"),
                "name": os.path.basename(path).replace(".gguf", "") if path else str(a.get("id")),
                "scale": _to_float(a.get("scale", 0)),
            })
        return loras
    except Exception:
        # endpoint absent or no adapters compiled in — treat as "none applied"
        return []


def fetch_vllm_props(port):
    """vLLM has no /props; synthesize the same shape from /v1/models."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", headers={"User-Agent": "curl"})
        resp = scrape_open(req, timeout=3)
        data = json.loads(read_capped(resp))
        # Same reasoning as fetch_llama_props: validate shape before indexing. `data` being a
        # list, or `data["data"]` holding non-objects, would raise AttributeError here and take
        # the whole scrape down with it.
        if not isinstance(data, dict):
            return {"_error": "models payload was not an object"}
        entries = data.get("data")
        entries = entries if isinstance(entries, list) else []
        m = next((e for e in entries if isinstance(e, dict)), {})
        return {
            "alias": str(m.get("id", "") or ""),
            "model_path": str(m.get("root", "") or ""),
            "n_ctx": _to_int(m.get("max_model_len", 0)),
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
# {port: (last_pf_tok, last_pf_time, held_rate, last_change_monotonic)}
_VLLM_PP = {}
_PP_HOLD_MAX_S = 60.0     # a held prefill rate with no new burst behind it is stale, not current
# spec-decode acceptance burst-hold: {port: (last_draft_total, held_pct)}
_VLLM_SPEC = {}

# Generic burst-hold for values that only exist WHILE the engine is working: {(port, key): (v, t)}
_HOLD = {}


def hold_value(port, key, value, now=None):
    """Hold the last live reading of a bursty value, with its age.

    llama.cpp publishes `prompt_tokens_seconds` / `predicted_tokens_seconds` as instantaneous
    gauges that read 0 whenever the model is idle between generations, and vLLM's latency
    histograms stop moving for the same reason. Rendering that as "0 tok/s" or "— ms" makes a
    box that just answered a request look dead. Rendering the last burst FOREVER is the opposite
    failure — the one `_PP_HOLD_MAX_S` already exists to prevent — so the hold expires, and until
    it does the caller gets the age so the page can say *when* the number was true.

    `value` falsy (0 or None) means "no fresh reading this poll". Returns (value, age_s) where
    age_s is 0.0 while live and None when nothing is being held.
    """
    now = now or time.time()
    k = (port, key)
    if value:
        _HOLD[k] = (value, now)
        return value, 0.0
    held = _HOLD.get(k)
    if not held:
        return (0.0 if value == 0 else None), None
    age = now - held[1]
    if age > _PP_HOLD_MAX_S:
        del _HOLD[k]
        return (0.0 if value == 0 else None), None
    return held[0], age


def drop_holds(port):
    """Forget a port's held values — called when it goes down.

    Without this, a server that is restarted onto a different model comes back idle and inherits
    the PREVIOUS model's throughput and latency numbers, which is the most misleading state the
    dashboard could possibly be in.
    """
    for k in [k for k in _HOLD if k[0] == port]:
        del _HOLD[k]


def fetch_vllm_endpoint(port):
    """vLLM flavor of fetch_endpoint: same output shape, vllm: metric names.

    vLLM exports cumulative counters (prompt_tokens_total / generation_tokens_total),
    so PP/TG tok-s are derived from the delta since the previous poll (first
    poll after a restart reads 0 — settles by the second sample).
    """
    metrics = fetch_llama_metrics(port)  # generic Prometheus parse works for vllm: names too
    props = fetch_vllm_props(port)
    up = "_error" not in props
    if not up:
        drop_holds(port)
    n_ctx = props.get("n_ctx", 0) if up else 0
    kv_ratio = prompt_total = gen_total = running = waiting = 0.0
    pf_tok = pf_time = 0.0
    spec_acc = spec_draft = None   # spec-decode counters absent unless the server runs MTP/draft
    # Latency histograms: (Σseconds, #observations) pairs. vLLM is the only one of the two engines
    # that publishes these at all — see the llama.cpp path for why that half stays empty.
    lat = {k: [0.0, 0.0] for k in ("ttft", "itl", "queue")}
    _LAT_SERIES = (
        ("ttft", "vllm:time_to_first_token_seconds"),
        ("itl", "vllm:time_per_output_token_seconds"),
        ("queue", "vllm:request_queue_time_seconds"),
    )
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
            elif k.startswith("vllm:num_requests_waiting"):
                waiting += v
            elif k.startswith("vllm:request_prefill_kv_computed_tokens_sum"):
                pf_tok += v
            elif k.startswith("vllm:request_prefill_time_seconds_sum"):
                pf_time += v
            elif k.startswith("vllm:spec_decode_num_accepted_tokens_total"):
                spec_acc = (spec_acc or 0.0) + v
            elif k.startswith("vllm:spec_decode_num_draft_tokens_total"):
                spec_draft = (spec_draft or 0.0) + v
            else:
                for name, prefix in _LAT_SERIES:
                    # Exact suffix match only. `_bucket` shares the prefix and is a per-`le` count,
                    # so a prefix match would fold bucket counts into the observation count and
                    # divide the latency sum by a number several times too large.
                    if k == prefix + "_sum":
                        lat[name][0] += v
                    elif k == prefix + "_count":
                        lat[name][1] += v
    now = time.time()
    hist = _VLLM_HIST.setdefault(port, [])
    # server restart resets counters -> drop stale history so rates don't go negative-then-zero
    if hist and (prompt_total < hist[-1][1] or gen_total < hist[-1][2]):
        hist.clear()
    hist.append((now, prompt_total, gen_total, spec_acc or 0.0, spec_draft or 0.0,
                 lat["ttft"][0], lat["ttft"][1], lat["itl"][0], lat["itl"][1],
                 lat["queue"][0], lat["queue"][1]))
    del hist[: max(0, len(hist) - 120)]   # bound memory (~4 min at 2s polls)

    def _windowed_rate(idx, window):
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

    decode_tps = _windowed_rate(2, _DECODE_WINDOW_S)

    # Client-visible latency, from the histogram (Σseconds, #observations) pairs.
    #
    # Δsum/Δcount over a window is the mean latency of the requests that FINISHED in that window.
    # The lifetime sum/count is not that — it is every request since the server started, so a box
    # that was hammered this morning would still show this morning's TTFT tonight. When no request
    # completed in the window there is no current answer, so the value is held (with its age) and
    # then expires, rather than being backfilled with a lifetime average dressed up as "now".
    # The window is much longer than the decode window: completions are sparse events, not a rate.
    _LAT_WINDOW_S = 30.0
    _LAT_IDX = {"ttft": (5, 6), "itl": (7, 8), "queue": (9, 10)}

    def _windowed_mean_ms(name):
        s_idx, c_idx = _LAT_IDX[name]
        cur_sum, cur_cnt = lat[name]
        if cur_cnt <= 0:
            return None
        ref = None
        for s in hist:
            if now - s[0] <= _LAT_WINDOW_S:
                break
            ref = s
        ref = ref or hist[0]
        d_cnt = cur_cnt - ref[c_idx]
        d_sum = cur_sum - ref[s_idx]
        if d_cnt <= 0 or d_sum < 0:
            return None                       # nothing completed in the window — hold or expire
        return 1000.0 * d_sum / d_cnt

    latency = {}
    for _name in ("ttft", "itl", "queue"):
        # Freshness is "did a request complete since the last poll", NOT "does the window still
        # contain one". A 30s trailing window keeps yielding a mean for 30s after the last
        # completion, so keying the age off the window would report "0s ago" half a minute after
        # the box went quiet — the exact stale-reading-presented-as-current failure this file
        # keeps guarding against. The mean itself is still the right number to show; only its
        # age changes. Once nothing has completed for `_PP_HOLD_MAX_S` it expires entirely.
        _c_idx = _LAT_IDX[_name][1]
        _advanced = len(hist) >= 2 and hist[-1][_c_idx] > hist[-2][_c_idx]
        _val, _age = hold_value(port, "lat_" + _name,
                                _windowed_mean_ms(_name) if _advanced else None, now)
        latency[_name + "_ms"] = round(_val, 1) if _val else None
        latency[_name + "_age_s"] = round(_age, 1) if _age is not None else None

    # prompt: burst-hold from the prefill-time histogram pair.
    # The hold is bounded. Holding the last burst FOREVER meant one 1,000 tok/s prefill was still
    # displayed as the current rate an hour into an idle box — a confidently wrong number, which
    # is the worst thing a metrics dashboard can show.
    lt, lp, held, changed_at = _VLLM_PP.get(port, (None, None, 0.0, 0.0))
    now_pp = time.time()
    if lt is not None and (pf_tok < lt or pf_time < lp):
        lt = lp = None                                     # server restart -> counter reset
    if lt is None:
        held = (pf_tok / pf_time) if pf_time > 0 else 0.0  # seed: lifetime average
        changed_at = now_pp
    elif pf_tok > lt and pf_time > lp:
        held = (pf_tok - lt) / (pf_time - lp)              # rate of the latest burst
        changed_at = now_pp
    elif now_pp - changed_at > _PP_HOLD_MAX_S:
        held = 0.0                                         # no prefill for a while: idle, not 1000 t/s
    _VLLM_PP[port] = (pf_tok, pf_time, held, changed_at)
    prompt_tps = held

    # MTP/spec-decode acceptance % — burst-hold like PP (decode bursts are short; 0 between
    # generations is not a valid answer). Rate of the latest burst = Δaccepted/Δdrafted over
    # the recent window; seeded with the lifetime ratio. None when the server runs no drafter.
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
    return {
        "status": "up" if up else "down",
        "metrics": metrics,
        "props": props,
        "loras": [],
        "derived": {
            "prompt_tps": round(prompt_tps, 1),
            "decode_tps": round(decode_tps, 1),
            # vLLM's PP/TG are already burst-held above, so they are never a stale-gauge zero the
            # way llama.cpp's are; the age fields exist so both engines hand the page one shape.
            "prompt_tps_age_s": None,
            "decode_tps_age_s": None,
            "spec_accept_pct": spec_accept_pct,
            "n_ctx": n_ctx,
            "ctx_used_tokens": round(kv_ratio * n_ctx),
            "ctx_fill_pct": round(kv_ratio * 100, 1),
            "n_loras_active": 0,
            "n_loras": 0,
            "requests_processing": running,
            "requests_waiting": waiting if up else 0,
            "busy_slots_per_decode": None,   # llama.cpp-only counter
            **latency,
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
    if not up:
        drop_holds(port)
    loras = fetch_llama_loras(port) if up else []
    active_loras = [l for l in loras if (l.get("scale") or 0) > 0]
    n_ctx = props.get("n_ctx", 0) if isinstance(props, dict) else 0
    kv_ratio = metrics.get("llamacpp:kv_cache_usage_ratio", 0) if isinstance(metrics, dict) else 0

    # llama.cpp's two rate gauges are instantaneous: they read 0 the moment a generation ends, so
    # the panel used to flip to "0.0 tok/s" between requests on a perfectly healthy server. Hold
    # the last live reading with its age (bounded, then expires) — the same treatment the vLLM path
    # already gives its prefill rate. The age is what keeps it honest: "87 tok/s · 4s ago" is a
    # different claim from "87 tok/s", and only one of them is true on an idle box.
    pp_raw = metrics.get("llamacpp:prompt_tokens_seconds", 0) if up else 0
    tg_raw = metrics.get("llamacpp:predicted_tokens_seconds", 0) if up else 0
    prompt_tps, pp_age = hold_value(port, "pp", pp_raw if up else 0)
    decode_tps, tg_age = hold_value(port, "tg", tg_raw if up else 0)
    return {
        "status": "up" if up else "down",
        "metrics": metrics,
        "props": props,
        "loras": loras,
        "derived": {
            "prompt_tps": prompt_tps,
            "decode_tps": decode_tps,
            "prompt_tps_age_s": round(pp_age, 1) if pp_age else pp_age,
            "decode_tps_age_s": round(tg_age, 1) if tg_age else tg_age,
            "n_ctx": n_ctx,
            "ctx_used_tokens": metrics.get("llamacpp:kv_cache_tokens", round(kv_ratio * n_ctx)) if up else 0,
            "ctx_fill_pct": round(kv_ratio * 100, 1) if up else 0,
            "n_loras_active": len(active_loras),
            "n_loras": len(loras),
            "requests_processing": metrics.get("llamacpp:requests_processing", 0) if up else 0,
            # Deferred = accepted but with no free slot, i.e. the same "queue behind a full
            # engine" state vLLM calls num_requests_waiting. One key, so the page has one path.
            "requests_waiting": metrics.get("llamacpp:requests_deferred", 0) if up else 0,
            # Mean busy slots per decode step: how well the batch is being filled.
            "busy_slots_per_decode": metrics.get("llamacpp:n_busy_slots_per_decode") if up else None,
            # llama.cpp's exporter has eleven series and none of them is a latency histogram
            # (tools/server: prompt/predicted token+seconds totals, n_decode_total, n_tokens_max,
            # the two rate gauges, requests_processing/deferred, n_busy_slots_per_decode). TTFT
            # and inter-token latency are therefore NOT derivable from a scrape here — 1000/tok-s
            # would look like ITL but is an aggregate across every batched slot, not what any one
            # client waited. Measuring these needs a tap in the request path, not this scraper.
            "ttft_ms": None, "ttft_age_s": None,
            "itl_ms": None, "itl_age_s": None,
            "queue_ms": None, "queue_age_s": None,
        },
    }


def check_port(host, port):
    """Check if a TCP port is accepting connections."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/", headers={"User-Agent": "curl"})
        resp = scrape_open(req, timeout=2)
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


_NET_CACHE = {"t": 0.0, "data": None}
_NET_MIN_INTERVAL_S = 1.0   # recompute rates at most once a second; see get_network()


def get_network():
    """Per-physical-NIC RX/TX rates in bytes/s (delta since last poll) + lifetime totals.

    The rate snapshot is cached for _NET_MIN_INTERVAL_S so the numbers don't depend on HOW MANY
    clients are polling. The counter deltas live in one process-global (_NET_LAST) and used to be
    consumed by whichever request arrived first: open a second dashboard tab and it would read
    ~0 B/s (or a spike over a millisecond-wide interval) because the first tab had just reset the
    baseline. Every caller inside the window now gets the same snapshot, measured over a real
    interval.
    """
    now = time.time()
    cached = _NET_CACHE["data"]
    if cached is not None and (now - _NET_CACHE["t"]) < _NET_MIN_INTERVAL_S:
        return cached
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
    result = {"ifaces": ifaces, "rx_bps": round(total_rx_bps), "tx_bps": round(total_tx_bps)}
    _NET_CACHE["t"], _NET_CACHE["data"] = now, result
    return result


# LAN metadata — PASSIVE only (no probing/scanning): the kernel ARP table (`ip neigh`,
# devices this box has actually exchanged packets with) + established TCP connections
# (`ss -tn`). Names resolved from /etc/hosts, never live DNS. Cached 10s so the 2s
# dashboard poll stays cheap.
_LAN_CACHE = {"t": 0.0, "data": None}
def _classify_addr(raw):
    """('loopback' | 'private' | 'public' | None) for an address from `ss`, either family.

    Replaces a string-prefix table whose "172.2" entry matched all of 172.20-172.255, so public
    addresses such as 172.217.x.x were counted and displayed as LAN peers. RFC1918's middle block
    is 172.16.0.0/12 — a range that simply cannot be expressed as a string prefix. The prefix
    table also skipped IPv6 outright; `ipaddress` handles both families and gets ULA and
    link-local right for free.
    """
    try:
        addr = ipaddress.ip_address(raw.strip("[]"))
    except ValueError:
        return None
    if addr.is_loopback or addr.is_unspecified:
        return "loopback"
    return "private" if addr.is_private else "public"


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


def get_lan():
    now = time.time()
    if _LAN_CACHE["data"] is not None and now - _LAN_CACHE["t"] < 10:
        return _LAN_CACHE["data"]
    names = _hosts_names()
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
            kind = _classify_addr(pip)
            if kind is None or kind == "loopback":
                continue                      # unparseable, or this box talking to itself
            pip = pip.strip("[]")             # normalise the bracketed IPv6 form for display
            if kind == "private":
                e = lan_peers.setdefault(pip, {"ip": pip, "name": names.get(pip, ""),
                                               "conns": 0, "ports": set()})
                e["conns"] += 1
                # the service side of the socket is the low/registered port
                for prt in (lport, pport):
                    if prt.isdigit() and int(prt) < 30000:
                        e["ports"].add(int(prt))
            else:
                wan_conns += 1
    except Exception:
        pass
    peers = sorted(lan_peers.values(), key=lambda p: -p["conns"])
    for p in peers:
        p["ports"] = sorted(p["ports"])[:6]
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
    def _send_cors(self):
        """Emit an allowlisted CORS origin, or none. Always Vary: Origin so the response is not
        cached under one origin and replayed to another."""
        self.send_header("Vary", "Origin")
        allowed = cors_origin(self.headers.get("Origin"))
        if allowed:
            self.send_header("Access-Control-Allow-Origin", allowed)

    def do_GET(self):
        # Reject disallowed cross-origin requests BEFORE doing any work. Refusing only at
        # response-writing time still ran the full gather — every nvidia-smi call and upstream
        # scrape — on the single server thread. A hostile page cannot read the JSON, but it could
        # sit in a loop calling fetch(..., {mode:"no-cors"}) and keep the dashboard busy.
        origin = self.headers.get("Origin")
        if (origin and not cors_origin(origin)) or \
           self.headers.get("Sec-Fetch-Site") == "cross-site":
            body = b"403 - cross-origin request refused\n"
            self.send_response(403)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(body)
            return
        if not host_allowed(self.headers.get("Host")):
            body = (b"403 - unexpected Host header (DNS-rebinding guard).\n"
                    b"Reach this server as localhost/127.0.0.1, or set "
                    b"FLEET_METRICS_ALLOWED_HOSTS=your.host.name\n")
            self.send_response(403)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/metrics" or self.path == "/api/metrics":
            if FIXTURE:
                payload = json.load(open(os.path.expanduser(FIXTURE)))
            else:
                payload = self._gather()
            body = json.dumps(_json_safe(payload), indent=2, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/models":
            # loadable-model registry for the dashboard MODEL LIBRARY (edit models-registry.json)
            try:
                reg = os.environ.get("MODELS_REGISTRY",
                                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "models-registry.json"))
                body = open(os.path.expanduser(reg), "rb").read()
                self.send_response(200)
            except Exception as e:
                body = json.dumps({"_error": str(e)}).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/", "/index.html"):
            # Serve the dashboard from this origin so its fetches are same-origin and no CORS
            # grant is needed. Fixed path next to this file — nothing from the request reaches
            # the filesystem, so there is no traversal surface here.
            try:
                page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
                body = open(page, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            except OSError:
                body = b"index.html not found next to fleet-metrics.py"
                self.send_response(404)
                self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"404 - use / or /metrics")

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
            req = urllib.request.Request(f"http://127.0.0.1:{port}/thoughts", headers={"User-Agent": "curl"})
            with scrape_open(req, timeout=1.0) as r:
                data = json.loads(read_capped(r).decode("utf-8", "replace"))
        except Exception:
            return []
        # Shape-check like every other upstream. This one was exempt, so a tap returning
        # {"streams":[null]} reached the frontend and threw while reading `s.updated` — marking
        # the WHOLE dashboard offline even though GPU and worker collection had succeeded.
        if not isinstance(data, dict):
            return []
        streams = data.get("streams")
        if not isinstance(streams, list):
            return []
        clean = []
        for s in streams[:32]:                       # a tap cannot conjure unbounded panels
            if not isinstance(s, dict):
                continue
            clean.append({
                "id": str(s.get("id", "") or ""),
                "phase": str(s.get("phase", "") or ""),
                "text": str(s.get("text", "") or ""),
                "started": _to_float(s.get("started", 0)),
                "updated": _to_float(s.get("updated", 0)),
                "tokens": _to_int(s.get("tokens", 0)),
            })
        return clean

    def _gather(self):
        # Primary worker — keep the llama_8001 key shape, now also carrying
        # loras + derived fields so the worker panel can show ctx-fill / LoRAs too.
        worker_port = resolve_worker_port()
        worker = fetch_endpoint(worker_port)
        secondaries = []
        for d in SECONDARIES:
            snap = fetch_endpoint(d["port"])
            snap["name"] = d["name"]
            snap["port"] = d["port"]
            snap["model"] = d["model"]
            secondaries.append(snap)
        return {
            "timestamp": time.time(),
            "gpus": attach_gpu_procs(parse_nvidia_smi()),
            "worker_port": worker_port,
            "llama_8001": worker,
            "secondaries": secondaries,
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


class _HTTPServerV6(HTTPServer):
    """HTTPServer is AF_INET only, so FLEET_METRICS_BIND=::1 died with an address-family error."""
    address_family = socket.AF_INET6


def _make_server():
    try:
        v6 = isinstance(ipaddress.ip_address(BIND.strip("[]")), ipaddress.IPv6Address)
    except ValueError:
        v6 = False              # a name: let getaddrinfo via AF_INET decide, as before
    cls = _HTTPServerV6 if v6 else HTTPServer
    return cls((BIND.strip("[]"), PORT), Handler)


if __name__ == "__main__":
    print(f"fleet-metrics serving on {BIND}:{PORT}/metrics")
    if BIND not in ("127.0.0.1", "localhost", "::1"):
        print(f"  warning: bound to {BIND} — /metrics is reachable off-box and has no auth")
    _make_server().serve_forever()
