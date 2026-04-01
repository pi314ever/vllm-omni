"""Pytest: XPU sin/cos accuracy for RoPE-range inputs across dtypes and shapes.

Demonstrates that torch.sin()/torch.cos() on XPU produce incorrect values
for large inputs in certain dtypes or after certain tensor operations
(transpose, reshape) that affect strides/contiguity.

Usage:
    pytest test_xpu_sincos.py -v
"""

import math

import pytest
import torch

DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
DTYPE_IDS = ["bf16", "fp16", "fp32", "fp64"]

# Shapes that match LTX2 RoPE computation at different stages
SHAPES = [
    "flat_1024",  # [1024] — simple 1D
    "audio_pre_flatten",  # [1, 1, 1, 1024] — before transpose/flatten
    "audio_post_flatten",  # [1, 1, 1024] — after flatten
    "audio_reshaped",  # [1, 1, 32, 32] — after reshape to heads
    "audio_swapped",  # [1, 32, 1, 32] — after swapaxes (non-contiguous!)
    "video_3d_grid",  # [1, 1024, 3, 342] — video before transpose
    "video_transposed",  # [1, 1024, 342, 3] — video after transpose (non-contiguous)
]


def _rope_freqs_flat(device: str, dtype: torch.dtype) -> torch.Tensor:
    """Generate RoPE frequencies matching LTX2 config (theta=10000, dim=2048)."""
    theta = 10000.0
    steps = 1024
    pow_indices = torch.pow(
        theta,
        torch.linspace(0.0, 1.0, steps, dtype=torch.float64, device=device),
    )
    return (pow_indices * math.pi / 2.0).to(dtype=dtype)


def _make_shaped_freqs(shape_id: str, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Build a freq tensor in the given shape, mimicking the LTX2 RoPE pipeline."""
    base = _rope_freqs_flat(device, dtype)  # [1024]

    if shape_id == "flat_1024":
        return base

    if shape_id == "audio_pre_flatten":
        # Simulate: grid [1,1,1,1] * freqs [1024] -> [1,1,1,1024]
        grid_val = torch.tensor([[[[-0.9995]]]], device=device, dtype=dtype)
        return grid_val * base

    if shape_id == "audio_post_flatten":
        t = _make_shaped_freqs("audio_pre_flatten", device, dtype)
        return t.transpose(-1, -2).flatten(2)  # [1, 1, 1024]

    if shape_id == "audio_reshaped":
        t = _make_shaped_freqs("audio_post_flatten", device, dtype)
        return t.reshape(1, 1, 32, 32)

    if shape_id == "audio_swapped":
        t = _make_shaped_freqs("audio_reshaped", device, dtype)
        return t.swapaxes(1, 2)  # [1, 32, 1, 32] — NOT contiguous

    if shape_id == "video_3d_grid":
        grid = torch.randn(1, 1024, 3, device=device, dtype=dtype) * 0.1
        steps_per_dim = 342
        freq_vec = base[:steps_per_dim]
        return (grid.unsqueeze(-1) * 2 - 1) * freq_vec

    if shape_id == "video_transposed":
        t = _make_shaped_freqs("video_3d_grid", device, dtype)
        return t.transpose(-1, -2)  # non-contiguous

    raise ValueError(f"Unknown shape_id: {shape_id}")


def _cpu_reference(t: torch.Tensor):
    """Compute ground truth sin/cos on CPU in float64."""
    t_cpu = t.double().cpu()
    return t_cpu.cos(), t_cpu.sin()


@pytest.fixture(scope="module")
def xpu_device():
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")
    return "xpu:0"


# ---------------------------------------------------------------------------
# Test: sin/cos values must be bounded in [-1, 1]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_sincos_bounded(shape, dtype, xpu_device):
    """sin/cos outputs must have absolute value <= 1 regardless of shape/dtype."""
    try:
        freqs = _make_shaped_freqs(shape, xpu_device, dtype)
    except RuntimeError:
        pytest.skip(f"dtype {dtype} unsupported for shape {shape}")

    cos_vals = freqs.cos()
    sin_vals = freqs.sin()

    cos_absmax = float(cos_vals.float().abs().max())
    sin_absmax = float(sin_vals.float().abs().max())

    assert cos_absmax <= 1.0 + 1e-3, (
        f"cos({dtype}, {shape}) absmax={cos_absmax:.6e} > 1.0  "
        f"[contiguous={freqs.is_contiguous()}, shape={list(freqs.shape)}, strides={freqs.stride()}]"
    )
    assert sin_absmax <= 1.0 + 1e-3, (
        f"sin({dtype}, {shape}) absmax={sin_absmax:.6e} > 1.0  "
        f"[contiguous={freqs.is_contiguous()}, shape={list(freqs.shape)}, strides={freqs.stride()}]"
    )


# ---------------------------------------------------------------------------
# Test: accuracy vs CPU float64 reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
@pytest.mark.parametrize("shape", SHAPES)
def test_cos_accuracy(shape, dtype, xpu_device):
    """cos() on XPU should match CPU float64 reference (fp32/fp64 only)."""
    try:
        freqs = _make_shaped_freqs(shape, xpu_device, dtype)
    except RuntimeError:
        pytest.skip(f"dtype {dtype} unsupported for shape {shape}")

    cos_ref, _ = _cpu_reference(freqs)
    cos_xpu = freqs.cos()

    err = (cos_xpu.double().cpu() - cos_ref).abs()
    n_broken = int((err > 1.0).sum())

    assert n_broken == 0, (
        f"cos({dtype}, {shape}): {n_broken}/{freqs.numel()} elements |err|>1  "
        f"max_err={float(err.max()):.6e}  absmax={float(cos_xpu.float().abs().max()):.6e}  "
        f"contiguous={freqs.is_contiguous()}  strides={freqs.stride()}"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
@pytest.mark.parametrize("shape", SHAPES)
def test_sin_accuracy(shape, dtype, xpu_device):
    """sin() on XPU should match CPU float64 reference (fp32/fp64 only)."""
    try:
        freqs = _make_shaped_freqs(shape, xpu_device, dtype)
    except RuntimeError:
        pytest.skip(f"dtype {dtype} unsupported for shape {shape}")

    _, sin_ref = _cpu_reference(freqs)
    sin_xpu = freqs.sin()

    err = (sin_xpu.double().cpu() - sin_ref).abs()
    n_broken = int((err > 1.0).sum())

    assert n_broken == 0, (
        f"sin({dtype}, {shape}): {n_broken}/{freqs.numel()} elements |err|>1  "
        f"max_err={float(err.max()):.6e}  absmax={float(sin_xpu.float().abs().max()):.6e}  "
        f"contiguous={freqs.is_contiguous()}  strides={freqs.stride()}"
    )


# ---------------------------------------------------------------------------
# Test: contiguous() before sin/cos as a potential workaround
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
@pytest.mark.parametrize("shape", SHAPES)
def test_contiguous_workaround(shape, dtype, xpu_device):
    """Calling .contiguous() before sin/cos should produce correct results."""
    try:
        freqs = _make_shaped_freqs(shape, xpu_device, dtype)
    except RuntimeError:
        pytest.skip(f"dtype {dtype} unsupported for shape {shape}")

    cos_ref, sin_ref = _cpu_reference(freqs)
    freqs_c = freqs.contiguous()
    cos_fix = freqs_c.cos()
    sin_fix = freqs_c.sin()

    cos_err = (cos_fix.double().cpu() - cos_ref).abs()
    sin_err = (sin_fix.double().cpu() - sin_ref).abs()

    assert int((cos_err > 1.0).sum()) == 0, f"contiguous cos({dtype}, {shape}): max_err={float(cos_err.max()):.6e}"
    assert int((sin_err > 1.0).sum()) == 0, f"contiguous sin({dtype}, {shape}): max_err={float(sin_err.max()):.6e}"


# ---------------------------------------------------------------------------
# Test: f64 upcast workaround
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_f64_workaround(shape, dtype, xpu_device):
    """Upcast to f64, compute sin/cos, downcast — should match CPU reference."""
    try:
        freqs = _make_shaped_freqs(shape, xpu_device, dtype)
    except RuntimeError:
        pytest.skip(f"dtype {dtype} unsupported for shape {shape}")

    cos_ref, sin_ref = _cpu_reference(freqs)
    cos_fix = freqs.double().cos().to(dtype)
    sin_fix = freqs.double().sin().to(dtype)

    cos_err = (cos_fix.double().cpu() - cos_ref).abs()
    sin_err = (sin_fix.double().cpu() - sin_ref).abs()

    assert int((cos_err > 1.0).sum()) == 0, f"f64 cos({dtype}, {shape}): max_err={float(cos_err.max()):.6e}"
    assert int((sin_err > 1.0).sum()) == 0, f"f64 sin({dtype}, {shape}): max_err={float(sin_err.max()):.6e}"
