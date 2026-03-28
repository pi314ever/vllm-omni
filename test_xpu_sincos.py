"""Pytest: XPU sin/cos accuracy for RoPE-range inputs across dtypes.

Demonstrates that torch.sin()/torch.cos() on XPU produce incorrect values
for large inputs in certain dtypes due to broken range-reduction, while
higher-precision computation works correctly.

Usage:
    pytest test_xpu_sincos.py -v
"""

import math

import pytest
import torch

DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
DTYPE_IDS = ["bf16", "fp16", "fp32", "fp64"]


def _rope_freqs(device: str = "cpu", dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Generate RoPE frequencies matching LTX2 config (theta=10000, dim=2048)."""
    theta = 10000.0
    steps = 1024
    pow_indices = torch.pow(
        theta,
        torch.linspace(0.0, 1.0, steps, dtype=torch.float64, device=device),
    )
    return (pow_indices * math.pi / 2.0).to(dtype=dtype)


@pytest.fixture(scope="module")
def cpu_reference():
    """Ground truth: float64 sin/cos computed on CPU."""
    freqs = _rope_freqs(device="cpu", dtype=torch.float64)
    return freqs.cos(), freqs.sin()


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cos_accuracy(dtype, cpu_reference):
    """cos() on XPU in the given dtype should match CPU float64 reference."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    cos_ref, _ = cpu_reference
    freqs_xpu = _rope_freqs(device="xpu:0", dtype=dtype)
    cos_xpu = freqs_xpu.cos()

    err = (cos_xpu.double().cpu() - cos_ref).abs()
    n_broken = int((err > 1.0).sum())
    max_err = float(err.max())

    assert n_broken == 0, (
        f"cos({dtype}): {n_broken}/1024 elements have |err|>1 vs CPU f64 reference "
        f"(max_err={max_err:.6e}, absmax={float(cos_xpu.abs().max()):.6e})"
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_sin_accuracy(dtype, cpu_reference):
    """sin() on XPU in the given dtype should match CPU float64 reference."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    _, sin_ref = cpu_reference
    freqs_xpu = _rope_freqs(device="xpu:0", dtype=dtype)
    sin_xpu = freqs_xpu.sin()

    err = (sin_xpu.double().cpu() - sin_ref).abs()
    n_broken = int((err > 1.0).sum())
    max_err = float(err.max())

    assert n_broken == 0, (
        f"sin({dtype}): {n_broken}/1024 elements have |err|>1 vs CPU f64 reference "
        f"(max_err={max_err:.6e}, absmax={float(sin_xpu.abs().max()):.6e})"
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_sincos_bounded(dtype):
    """sin() and cos() outputs must be in [-1, 1] regardless of input magnitude."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    freqs = _rope_freqs(device="xpu:0", dtype=dtype)
    cos_vals = freqs.cos()
    sin_vals = freqs.sin()

    cos_absmax = float(cos_vals.float().abs().max())
    sin_absmax = float(sin_vals.float().abs().max())

    assert cos_absmax <= 1.0 + 1e-3, (
        f"cos({dtype}) absmax={cos_absmax:.6e}, expected <= 1.0"
    )
    assert sin_absmax <= 1.0 + 1e-3, (
        f"sin({dtype}) absmax={sin_absmax:.6e}, expected <= 1.0"
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_f64_workaround(dtype, cpu_reference):
    """Workaround: upcast to f64, compute sin/cos, downcast — should match CPU reference."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    cos_ref, sin_ref = cpu_reference
    freqs_xpu = _rope_freqs(device="xpu:0", dtype=dtype)

    cos_fix = freqs_xpu.double().cos().to(dtype)
    sin_fix = freqs_xpu.double().sin().to(dtype)

    cos_err = (cos_fix.double().cpu() - cos_ref).abs()
    sin_err = (sin_fix.double().cpu() - sin_ref).abs()

    assert int((cos_err > 1.0).sum()) == 0, (
        f"f64-workaround cos({dtype}): still broken, max_err={float(cos_err.max()):.6e}"
    )
    assert int((sin_err > 1.0).sum()) == 0, (
        f"f64-workaround sin({dtype}): still broken, max_err={float(sin_err.max()):.6e}"
    )
