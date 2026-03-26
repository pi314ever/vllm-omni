# Debug memory logger for LTX2 memory leak investigation
# Session: efc1c2 - writes NDJSON to log file

import gc
import json
import os
import sys
import time
import traceback

import torch

_LOG_PATH = "./debug-efc1c2.log"
_SESSION_ID = "efc1c2"
_REQUEST_COUNTER = 0
_STEP_COUNTER = 0


def _get_device_mem():
    """Get XPU/CUDA memory stats. Returns dict with free, allocated, reserved."""
    result = {}
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            free, total = torch.xpu.mem_get_info()
            result["free_gib"] = round(free / (1024**3), 3)
            result["total_gib"] = round(total / (1024**3), 3)
            result["used_gib"] = round((total - free) / (1024**3), 3)
            if hasattr(torch.xpu, "memory_allocated"):
                result["allocated_gib"] = round(torch.xpu.memory_allocated() / (1024**3), 3)
            if hasattr(torch.xpu, "memory_reserved"):
                result["reserved_gib"] = round(torch.xpu.memory_reserved() / (1024**3), 3)
        elif torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            result["free_gib"] = round(free / (1024**3), 3)
            result["total_gib"] = round(total / (1024**3), 3)
            result["used_gib"] = round((total - free) / (1024**3), 3)
            result["allocated_gib"] = round(torch.cuda.memory_allocated() / (1024**3), 3)
            result["reserved_gib"] = round(torch.cuda.memory_reserved() / (1024**3), 3)
    except Exception as e:
        result["error"] = str(e)
    return result


def _tensor_summary(t):
    """Summarize a tensor's memory footprint."""
    if not isinstance(t, torch.Tensor):
        return {"type": str(type(t))}
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "nbytes_mib": round(t.nelement() * t.element_size() / (1024**2), 2),
    }


def _write_log(entry):
    """Append a single NDJSON line to the log file."""
    try:
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        with open(_LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass


def log_event(location, message, data=None, hypothesis_id=None, run_id=None):
    """Log a debug event with memory snapshot."""
    global _REQUEST_COUNTER, _STEP_COUNTER
    entry = {
        "sessionId": _SESSION_ID,
        "id": f"log_{int(time.time()*1000)}_{os.getpid()}",
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "pid": os.getpid(),
        "request_num": _REQUEST_COUNTER,
        "step_num": _STEP_COUNTER,
        "mem": _get_device_mem(),
    }
    if data:
        entry["data"] = data
    if hypothesis_id:
        entry["hypothesisId"] = hypothesis_id
    if run_id:
        entry["runId"] = run_id
    _write_log(entry)


def increment_request():
    global _REQUEST_COUNTER, _STEP_COUNTER
    _REQUEST_COUNTER += 1
    _STEP_COUNTER = 0


def increment_step():
    global _STEP_COUNTER
    _STEP_COUNTER += 1


def log_gc_stats(location):
    """Log garbage collector stats to detect reference retention."""
    counts = gc.get_count()
    log_event(
        location,
        "gc_stats",
        data={"gc_counts": list(counts), "gc_objects": len(gc.get_objects())},
        hypothesis_id="D",
    )
