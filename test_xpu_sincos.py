"""Isolated test: XPU sin/cos accuracy for large float32 inputs.

Demonstrates that torch.sin() and torch.cos() on XPU produce garbage
values for large float32 inputs due to a broken range-reduction algorithm,
while float64 (double) works correctly.

Usage:
    python test_xpu_sincos.py
"""

import torch
import math

def test_sincos_accuracy():
    assert torch.xpu.is_available(), "XPU not available"
    device = "xpu:0"

    # RoPE frequencies: theta^(linspace(0,1,1024)) * pi/2
    # Range: ~1.57 to ~15708
    theta = 10000.0
    steps = 1024
    pow_indices = torch.pow(
        theta,
        torch.linspace(0.0, 1.0, steps, dtype=torch.float64, device=device),
    )
    freqs_f64 = pow_indices * math.pi / 2.0  # float64
    freqs_f32 = freqs_f64.float()             # float32

    # Ground truth: float64 sin/cos (correct on both CPU and XPU)
    cos_ref = freqs_f64.cos()
    sin_ref = freqs_f64.sin()

    # Test: float32 sin/cos on XPU
    cos_f32 = freqs_f32.cos()
    sin_f32 = freqs_f32.sin()

    # float64 sin/cos on XPU (proposed fix)
    cos_fix = freqs_f32.double().cos().float()
    sin_fix = freqs_f32.double().sin().float()

    print("=" * 70)
    print("XPU sin/cos accuracy test")
    print("=" * 70)
    print(f"  freqs range: [{float(freqs_f32.min()):.2f}, {float(freqs_f32.max()):.2f}]")
    print(f"  freqs dtype: {freqs_f32.dtype}, device: {freqs_f32.device}")
    print()

    # Compare cos
    cos_err = (cos_f32.double() - cos_ref).abs()
    cos_fix_err = (cos_fix.double() - cos_ref).abs()
    print("--- cos ---")
    print(f"  f32 cos range:  [{float(cos_f32.min()):.6f}, {float(cos_f32.max()):.6f}]")
    print(f"  f32 cos absmax: {float(cos_f32.abs().max()):.6f}  (expected <= 1.0)")
    print(f"  f32 cos max_err vs f64 ref: {float(cos_err.max()):.6e}")
    print(f"  f32 cos elements with |err| > 1: {int((cos_err > 1).sum())} / {steps}")
    print()
    print(f"  fix cos range:  [{float(cos_fix.min()):.6f}, {float(cos_fix.max()):.6f}]")
    print(f"  fix cos absmax: {float(cos_fix.abs().max()):.6f}  (expected <= 1.0)")
    print(f"  fix cos max_err vs f64 ref: {float(cos_fix_err.max()):.6e}")
    print()

    # Compare sin
    sin_err = (sin_f32.double() - sin_ref).abs()
    sin_fix_err = (sin_fix.double() - sin_ref).abs()
    print("--- sin ---")
    print(f"  f32 sin range:  [{float(sin_f32.min()):.6f}, {float(sin_f32.max()):.6f}]")
    print(f"  f32 sin absmax: {float(sin_f32.abs().max()):.6e}  (expected <= 1.0)")
    print(f"  f32 sin max_err vs f64 ref: {float(sin_err.max()):.6e}")
    print(f"  f32 sin elements with |err| > 1: {int((sin_err > 1).sum())} / {steps}")
    print()
    print(f"  fix sin range:  [{float(sin_fix.min()):.6f}, {float(sin_fix.max()):.6f}]")
    print(f"  fix sin absmax: {float(sin_fix.abs().max()):.6f}  (expected <= 1.0)")
    print(f"  fix sin max_err vs f64 ref: {float(sin_fix_err.max()):.6e}")
    print()

    # Show worst offenders
    print("--- worst sin values (f32 on XPU) ---")
    worst_idx = sin_err.topk(min(5, steps)).indices
    for i in worst_idx:
        print(f"  freq={float(freqs_f32[i]):.2f}  "
              f"sin_f32={float(sin_f32[i]):.6e}  "
              f"sin_ref={float(sin_ref[i]):.6f}  "
              f"err={float(sin_err[i]):.6e}")
    print()

    print("--- worst cos values (f32 on XPU) ---")
    worst_idx = cos_err.topk(min(5, steps)).indices
    for i in worst_idx:
        print(f"  freq={float(freqs_f32[i]):.2f}  "
              f"cos_f32={float(cos_f32[i]):.6e}  "
              f"cos_ref={float(cos_ref[i]):.6f}  "
              f"err={float(cos_err[i]):.6e}")
    print()

    # Summary
    f32_broken = int((sin_err > 1).sum()) + int((cos_err > 1).sum())
    fix_broken = int((sin_fix_err > 1).sum()) + int((cos_fix_err > 1).sum())
    print("=" * 70)
    print(f"RESULT: f32 sin/cos has {f32_broken} / {steps * 2} values with |err| > 1")
    print(f"RESULT: f64 fix    has {fix_broken} / {steps * 2} values with |err| > 1")
    if f32_broken > 0 and fix_broken == 0:
        print("CONCLUSION: XPU float32 sin/cos is broken; float64 workaround fixes it")
    elif f32_broken == 0:
        print("CONCLUSION: XPU float32 sin/cos appears correct on this device")
    print("=" * 70)


if __name__ == "__main__":
    test_sincos_accuracy()
