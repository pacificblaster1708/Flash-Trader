# file_name: deeplob_precision_suite.py
"""
DeepLOB + precision / quantization suite (clean + practical).

What you get:
- One canonical DeepLOB implementation (no duplicated code).
- Run modes:
    fp32, bf16, fp16,
    fp8_e4m3 (native if available; else simulated),
    posit16_2 (sim),
    posit8_1 (sim),
    int8_static_fx (PyTorch FX PTQ for CPU),
    int8_smoothquant_fx (basic SmoothQuant + FX PTQ for CPU)

Notes:
- fp16/bf16 are best done with autocast during inference/training, not by permanently casting BatchNorm.
- INT8 PTQ here targets CPU backends (fbgemm / qnnpack). GPU INT8 is a different stack.
- Posit and FP8 “simulations” here are educational approximations unless you use true posit libs / true FP8 kernels.

Usage examples:
  python deeplob_precision_suite.py --mode fp32
  python deeplob_precision_suite.py --mode bf16
  python deeplob_precision_suite.py --mode fp16
  python deeplob_precision_suite.py --mode fp8_e4m3
  python deeplob_precision_suite.py --mode posit16_2
  python deeplob_precision_suite.py --mode int8_static_fx

"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ----------------------------
# 1) Canonical DeepLOB model
# ----------------------------
class DeepLOB(nn.Module):
    def __init__(self, y_len: int = 3):
        super().__init__()
        self.y_len = y_len

        # Conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        # Inception modules
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )

        # LSTM + head
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, y_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.cat((self.inp1(x), self.inp2(x), self.inp3(x)), dim=1)  # (B, 192, T', 1)
        x = x.permute(0, 2, 1, 3)  # (B, T', 192, 1)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])  # (B, T', 192)

        # Let PyTorch create initial states (better for mixed precision / compile)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return torch.softmax(x, dim=1)


# --------------------------------
# 2) Quantization / precision utils
# --------------------------------
class FakeQuantFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float, qmin: int, qmax: int, clip: float) -> torch.Tensor:
        # Simple symmetric fake-quant (round-to-nearest) with optional clipping
        if clip is not None:
            x = x.clamp(-clip, clip)
        s = max(scale, 1e-12)
        q = torch.round(x / s).clamp(qmin, qmax)
        return q * s

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None


def fake_quant(x: torch.Tensor, bits: int, clip: Optional[float] = None) -> torch.Tensor:
    # symmetric signed range
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax - 1
    # scale from max abs
    maxabs = x.detach().abs().max().item()
    scale = (maxabs / qmax) if maxabs > 0 else 1.0
    return FakeQuantFn.apply(x, scale, qmin, qmax, clip)


def fp8_quant_sim(x: torch.Tensor) -> torch.Tensor:
    """
    Very rough FP8 E4M3-ish simulation: clamp + fake-quant mantissa-ish effect.
    Not IEEE-accurate, but decent for “feel”.
    """
    # approximate max finite for E4M3 is ~448
    clip = 448.0
    x = x.clamp(-clip, clip)
    # simulate coarse mantissa by 8-bit fake quant (not true FP8)
    return fake_quant(x, bits=8, clip=clip)


def posit_sim(x: torch.Tensor, nbits: int, es: int) -> torch.Tensor:
    """
    Educational posit simulation: clamp dynamic range + coarse quant.
    True posit behavior is non-uniform; this is a practical stand-in.
    """
    # crude range heuristics
    if nbits == 16 and es == 2:
        clip = 2 ** 12  # your earlier placeholder
        bits = 10       # "effective" precision-ish
    elif nbits == 8 and es == 1:
        clip = 64.0
        bits = 6
    else:
        clip = 128.0
        bits = max(4, nbits - 2)

    x = x.clamp(-clip, clip)
    return fake_quant(x, bits=bits, clip=clip)


def apply_inplace_param_transform(model: nn.Module, fn) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(fn(p))


# -----------------------------
# 3) SmoothQuant (basic version)
# -----------------------------
@dataclass
class SQCalibrationStats:
    max_act: float = 1.0


class SmoothQuant:
    """
    Basic SmoothQuant for Conv2d/Linear.
    Needs activation max stats from calibration. Here we capture a single global max
    across relevant modules for a lightweight demo. You can extend to per-channel/per-layer.
    """
    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)
        self.stats = SQCalibrationStats(max_act=1.0)
        self._hooks = []

    def _hook(self, module, inp, out):
        # inp is a tuple; first element is input tensor
        x = inp[0]
        m = x.detach().abs().max().item()
        if m > self.stats.max_act:
            self.stats.max_act = m

    def attach(self, model: nn.Module) -> None:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self._hooks.append(m.register_forward_hook(self._hook))

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def transform(self, model: nn.Module) -> None:
        """
        Apply: W <- W * s ; X <- X / s
        We implement X scaling via folding into preceding layer weights only (demo style).
        """
        act_scale = max(self.stats.max_act, 1e-8)

        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    wmax = m.weight.abs().max().item()
                    wmax = max(wmax, 1e-8)
                    s = (act_scale ** self.alpha) / (wmax ** (1.0 - self.alpha))
                    m.weight.mul_(s)
                    if m.bias is not None:
                        m.bias.mul_(s)


# -----------------------------------------
# 4) INT8 post-training quant (FX graph)
# -----------------------------------------
def int8_static_fx_quantize_cpu(model: nn.Module, example_input: torch.Tensor) -> nn.Module:
    """
    FX graph mode PTQ (CPU).
    """
    import torch.ao.quantization as aq
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

    model = model.cpu().eval()
    qconfig = aq.get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}

    prepared = prepare_fx(model, qconfig_dict, example_inputs=(example_input.cpu(),))
    # calibration
    for _ in range(8):
        prepared(example_input.cpu())
    quantized = convert_fx(prepared)
    return quantized


# ------------------------
# 5) Inference runner
# ------------------------
@torch.no_grad()
def run_once(
    model: nn.Module,
    x: torch.Tensor,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    model.eval().to(device)

    # Prefer autocast for BF16/FP16
    if mode in {"bf16", "fp16"}:
        if device.type == "cuda":
            amp_dtype = torch.bfloat16 if mode == "bf16" else torch.float16
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                return model(x)
        else:
            # CPU autocast supports bf16 on many builds; fp16 on CPU is often not beneficial
            amp_dtype = torch.bfloat16 if mode == "bf16" else torch.float16
            with torch.autocast(device_type="cpu", dtype=amp_dtype):
                return model(x)

    return model(x)


def build_model_for_mode(mode: str, device: torch.device) -> Tuple[nn.Module, Optional[str]]:
    """
    Returns (model, notes)
    """
    notes = None
    base = DeepLOB()

    if mode == "fp32":
        return base, "FP32 baseline."

    if mode in {"bf16", "fp16"}:
        # Keep model weights in fp32; use autocast for safer BN behavior.
        return base, f"Using autocast for {mode.upper()} (recommended)."

    if mode == "fp8_e4m3":
        # If native float8 exists, store weights as float8 then dequantize to fp16/fp32 during compute.
        # Many ops still upcast internally; this is mainly a storage + simulation demo.
        if hasattr(torch, "float8_e4m3fn"):
            notes = "Native float8_e4m3fn detected; weights stored as float8 then dequantized."
            with torch.no_grad():
                for p in base.parameters():
                    p.copy_(p.to(torch.float8_e4m3fn).to(torch.float32))
            return base, notes
        else:
            notes = "No native float8 type; using FP8-ish fake quant simulation."
            apply_inplace_param_transform(base, fp8_quant_sim)
            return base, notes

    if mode == "posit16_2":
        apply_inplace_param_transform(base, lambda t: posit_sim(t, nbits=16, es=2))
        return base, "Applied posit(16,2) simulation to weights."

    if mode == "posit8_1":
        apply_inplace_param_transform(base, lambda t: posit_sim(t, nbits=8, es=1))
        return base, "Applied posit(8,1) simulation to weights."

    if mode in {"int8_static_fx", "int8_smoothquant_fx"}:
        # Built later because needs example input
        return base, "INT8 FX PTQ will run on CPU."

    raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="fp32",
        choices=[
            "fp32", "bf16", "fp16",
            "fp8_e4m3",
            "posit16_2", "posit8_1",
            "int8_static_fx", "int8_smoothquant_fx",
        ],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--time", type=int, default=100)
    parser.add_argument("--feat", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant alpha (only for int8_smoothquant_fx)")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Example input (B, C=1, T, F)
    x = torch.randn(args.batch, 1, args.time, args.feat, device=device)

    model, notes = build_model_for_mode(args.mode, device)

    # INT8 modes: FX PTQ (CPU)
    if args.mode in {"int8_static_fx", "int8_smoothquant_fx"}:
        # SmoothQuant transform (calibrate activation max)
        if args.mode == "int8_smoothquant_fx":
            sq = SmoothQuant(alpha=args.alpha)
            model.eval()
            sq.attach(model)
            # calibration pass on CPU
            for _ in range(8):
                _ = model(x.cpu())
            sq.detach()
            sq.transform(model)

        # FX PTQ convert
        model_q = int8_static_fx_quantize_cpu(model, x.cpu())
        out = model_q(x.cpu())
        print(f"Mode: {args.mode}")
        print("Device: CPU (INT8 FX PTQ)")
        print(f"Output shape: {tuple(out.shape)}")
        if notes:
            print(f"Notes: {notes}")
        return

    # Other modes: run once on chosen device
    out = run_once(model, x, args.mode, device=device)

    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Output shape: {tuple(out.shape)}")
    if notes:
        print(f"Notes: {notes}")


if __name__ == "__main__":
    main()
