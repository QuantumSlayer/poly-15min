# ---- hub_debug.py (or put in market_lib.py) --------------------------------
import os, json, asyncio, glob, math
from datetime import datetime
from typing import Any, Dict

def _safe(obj: Any):
    """
    Recursively convert arbitrary Python objects (incl. numpy, sets, deques,
    dicts with non-string keys, etc.) into JSON-serializable structures.
    """
    # Fast path
    if obj is None or isinstance(obj, (bool, int, float, str)):
        # Normalize NaN/Inf to strings to avoid JSON errors
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return str(obj)
        return obj

    # Common containers
    if isinstance(obj, (list, tuple, set)):
        return [_safe(x) for x in obj]
    if hasattr(obj, 'copy') and obj.__class__.__name__ == 'deque':
        return [_safe(x) for x in list(obj)]

    # Dicts (make sure keys are strings)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                ks = str(k)
            except Exception:
                ks = repr(k)
            out[ks] = _safe(v)
        return out

    # Numpy
    try:
        import numpy as np  # noqa
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # Dataclasses (by attribute dict)
    if hasattr(obj, "__dict__"):
        return {k: _safe(v) for k, v in vars(obj).items()
                if not callable(v) and not k.startswith("_")}

    # Fallback: string
    try:
        return str(obj)
    except Exception:
        return f"<unserializable:{type(obj).__name__}>"

def _hub_snapshot(hub) -> Dict[str, Any]:
    """
    Pick up *everything* in the hub that isn't a method/private attr.
    You can tailor the allow/deny list here if needed.
    """
    snap = {}
    for k, v in vars(hub).items():
        if k.startswith("_"):
            continue
        if callable(v):
            continue
        snap[k] = _safe(v)
    return snap

async def run_hub_debug_dumper(
    hub,
    out_dir: str = os.path.join(os.getcwd(), "data", "debug"),
    file_prefix: str = "hub",
    interval_sec: float = 1.0,
    rotate_mb: int = 50,          # rotate each ~50MB
):
    """
    Every `interval_sec` seconds, append one JSON line with the full hub snapshot.
    Files rotate when reaching ~`rotate_mb` MB.
    """
    os.makedirs(out_dir, exist_ok=True)
    rotate_bytes = int(rotate_mb * 1024 * 1024)

    # Determine next index
    existing = sorted(glob.glob(os.path.join(out_dir, f"{file_prefix}-*.jsonl")))
    if existing:
        try:
            last = os.path.basename(existing[-1])
            idx = int(last.split("-")[-1].split(".")[0]) + 1
        except Exception:
            idx = len(existing)
    else:
        idx = 0

    def _open_path(i: int) -> str:
        return os.path.join(out_dir, f"{file_prefix}-{i:05d}.jsonl")

    path = _open_path(idx)
    fh = open(path, "a", encoding="utf-8")
    bytes_written = 0

    try:
        while True:
            # Build snapshot
            ts = datetime.utcnow().isoformat() + "Z"
            payload = {"ts": ts, "hub": _hub_snapshot(hub)}
            line = json.dumps(payload, separators=(",", ":")) + "\n"

            # Rotate if needed
            if bytes_written + len(line) > rotate_bytes:
                try:
                    fh.flush(); fh.close()
                except Exception:
                    pass
                idx += 1
                path = _open_path(idx)
                fh = open(path, "a", encoding="utf-8")
                bytes_written = 0

            fh.write(line)
            bytes_written += len(line)

            # Flush occasionally for safety
            if bytes_written % (1 << 16) < len(line):  # ~every 64KB
                try:
                    fh.flush()
                except Exception:
                    pass

            await asyncio.sleep(max(0.05, float(interval_sec)))
    finally:
        try:
            fh.flush(); fh.close()
        except Exception:
            pass
