#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import onnx
import torch
import torch.nn as nn
from onnxsim import simplify


@dataclass
class TrialResult:
    trial_id: int
    cfg: dict
    onnx_path: str
    sim_onnx_path: str
    node_count: int
    trtexec_rc: int
    status: str
    log_path: str


class TinyRepro(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        n_attn: int,
        n_conv: int,
        use_state_unsqueeze: bool,
        use_rope_split: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_attn = n_attn
        self.n_conv = n_conv
        self.use_state_unsqueeze = use_state_unsqueeze
        self.use_rope_split = use_rope_split

        self.emb = nn.Embedding(4096, hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.conv_blocks = nn.ModuleList(
            [nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3) for _ in range(n_conv)]
        )
        self.proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def _attn_once(
        self, x: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # [B,T,H]
        b, t, _ = q.shape
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope_split:
            k1 = k[..., : self.head_dim // 2]
            k2 = k[..., self.head_dim // 2 :]
            k_rot = torch.cat((-k2, k1), dim=-1)
            q = q + k_rot

        k_all = torch.cat([past_k, k], dim=2)
        v_all = torch.cat([past_v, v], dim=2)
        score = torch.matmul(q, k_all.transpose(-2, -1)) * 0.5
        score = score + bias
        prob = torch.softmax(score, dim=-1)
        ctx = torch.matmul(prob, v_all)
        return self.out_proj(ctx.transpose(1, 2).reshape(b, t, self.hidden_dim))

    def forward(
        self,
        codes: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        attn_bias: torch.Tensor,
        conv_state: torch.Tensor,
        overlap: torch.Tensor,
    ) -> torch.Tensor:
        x = self.emb(codes).sum(dim=1)  # [B,4,H]
        for _ in range(self.n_attn):
            x = self._attn_once(x, past_k, past_v, attn_bias)

        if self.use_state_unsqueeze:
            s4 = conv_state.unsqueeze(1)  # [B,1,H,6]
            s4 = s4[:, :, :, :4]
            s = s4.mean(dim=3).squeeze(1).unsqueeze(1)  # [B,1,H]
        else:
            s = conv_state[:, :, :4].mean(dim=-1).unsqueeze(1)  # [B,1,H]
        x = x + s

        h = x.transpose(1, 2)  # [B,H,4]
        merged = torch.cat([conv_state[:, :, 2:], h], dim=2)
        y = merged[:, :, -6:]
        for conv in self.conv_blocks:
            y = conv(y)
            y = torch.sin(y) * 0.1 + y
            y = torch.cat([y[:, :, :1], y], dim=2)[:, :, :4]
        z = torch.cat([overlap, y[:, :, :3]], dim=2)
        wav = self.proj(z).clip(-1.0, 1.0)
        return wav


def run_cmd(cmd: Sequence[str], cwd: Path, log_file: Path) -> int:
    with log_file.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def contains(path: Path, text: str) -> bool:
    return text in path.read_text(encoding="utf-8", errors="ignore")


def export_trial(
    out_dir: Path, trial_id: int, cfg: dict, device: str = "cpu"
) -> tuple[Path, Path, int]:
    hidden_dim = int(cfg["hidden_dim"])
    num_heads = int(cfg["num_heads"])
    head_dim = hidden_dim // num_heads
    model = TinyRepro(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        n_attn=int(cfg["n_attn"]),
        n_conv=int(cfg["n_conv"]),
        use_state_unsqueeze=bool(cfg["use_state_unsqueeze"]),
        use_rope_split=bool(cfg["use_rope_split"]),
    ).to(device)
    model.eval()

    b, p = 1, 4
    inputs = (
        torch.randint(0, 4096, (b, 16, 4), dtype=torch.int64, device=device),
        torch.randn(b, num_heads, p, head_dim, device=device),
        torch.randn(b, num_heads, p, head_dim, device=device),
        torch.randn(b, 1, 4, p + 4, device=device),
        torch.randn(b, hidden_dim, 6, device=device),
        torch.randn(b, hidden_dim, 3, device=device),
    )
    dyn = {
        "codes": {0: "batch"},
        "past_k": {0: "batch", 2: "past"},
        "past_v": {0: "batch", 2: "past"},
        "attn_bias": {0: "batch", 3: "key_total"},
        "conv_state": {0: "batch"},
        "overlap": {0: "batch"},
        "wav": {0: "batch"},
    }
    onnx_path = out_dir / f"trial_{trial_id:03d}.onnx"
    sim_path = out_dir / f"trial_{trial_id:03d}.sim.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(onnx_path),
            input_names=["codes", "past_k", "past_v", "attn_bias", "conv_state", "overlap"],
            output_names=["wav"],
            dynamic_axes=dyn,
            opset_version=18,
            dynamo=False,
        )
    ms, ok = simplify(str(onnx_path))
    if not ok:
        raise RuntimeError(f"onnxsim failed for trial {trial_id}")
    onnx.save(ms, str(sim_path))
    node_count = len(ms.graph.node)
    return onnx_path, sim_path, node_count


def trtexec_cmd(image: str, sim_onnx: Path, engine: Path, hidden_dim: int, heads: int, head_dim: int) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{sim_onnx.parent}:/mnt",
        image,
        "/usr/src/tensorrt/bin/trtexec",
        "--onnx=/mnt/" + sim_onnx.name,
        "--saveEngine=/mnt/" + engine.name,
        "--bf16",
        "--inputIOFormats=int64:chw,bf16:chw,bf16:chw,bf16:chw,bf16:chw,bf16:chw",
        "--outputIOFormats=bf16:chw",
        f"--minShapes=codes:1x16x4,past_k:1x{heads}x1x{head_dim},past_v:1x{heads}x1x{head_dim},attn_bias:1x1x4x5,conv_state:1x{hidden_dim}x6,overlap:1x{hidden_dim}x3",
        f"--optShapes=codes:1x16x4,past_k:1x{heads}x4x{head_dim},past_v:1x{heads}x4x{head_dim},attn_bias:1x1x4x8,conv_state:1x{hidden_dim}x6,overlap:1x{hidden_dim}x3",
        f"--maxShapes=codes:8x16x4,past_k:8x{heads}x72x{head_dim},past_v:8x{heads}x72x{head_dim},attn_bias:8x1x4x76,conv_state:8x{hidden_dim}x6,overlap:8x{hidden_dim}x3",
        "--memPoolSize=workspace:2048",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search for <=N-node PyTorch-exported ONNX that triggers TensorRT Myelin tensor.cpp:852."
    )
    parser.add_argument("--out-root", default="/tmp/myelin_pytorch_search", help="Output directory")
    parser.add_argument("--max-nodes", type=int, default=50, help="Only run TRT for simplified ONNX <= this node count")
    parser.add_argument("--max-trials", type=int, default=64, help="Maximum number of trial configs to run")
    parser.add_argument("--image", default="nvcr.io/nvidia/tritonserver:26.02-py3", help="Docker image with trtexec")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    onnx_dir = out_root / "onnx"
    log_dir = out_root / "logs"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    hidden_dims = [64, 96, 128]
    num_heads_options = [8, 16]
    n_attn_options = [1, 2, 3, 4]
    n_conv_options = [1, 2, 3]
    use_state_unsqueeze_options = [False, True]
    use_rope_split_options = [False, True]

    all_cfgs = []
    for h, nh, na, nc, us, ur in itertools.product(
        hidden_dims,
        num_heads_options,
        n_attn_options,
        n_conv_options,
        use_state_unsqueeze_options,
        use_rope_split_options,
    ):
        if h % nh != 0:
            continue
        all_cfgs.append(
            {
                "hidden_dim": h,
                "num_heads": nh,
                "n_attn": na,
                "n_conv": nc,
                "use_state_unsqueeze": us,
                "use_rope_split": ur,
            }
        )

    results: list[TrialResult] = []
    for tid, cfg in enumerate(all_cfgs[: args.max_trials]):
        try:
            onnx_path, sim_path, node_count = export_trial(onnx_dir, tid, cfg)
        except Exception as e:
            fail_log = log_dir / f"trial_{tid:03d}.export.err.log"
            fail_log.write_text(str(e), encoding="utf-8")
            results.append(
                TrialResult(
                    trial_id=tid,
                    cfg=cfg,
                    onnx_path=str(onnx_dir / f"trial_{tid:03d}.onnx"),
                    sim_onnx_path=str(onnx_dir / f"trial_{tid:03d}.sim.onnx"),
                    node_count=-1,
                    trtexec_rc=-1,
                    status="export_error",
                    log_path=str(fail_log),
                )
            )
            continue

        if node_count > args.max_nodes:
            results.append(
                TrialResult(
                    trial_id=tid,
                    cfg=cfg,
                    onnx_path=str(onnx_path),
                    sim_onnx_path=str(sim_path),
                    node_count=node_count,
                    trtexec_rc=-1,
                    status="skip_too_large",
                    log_path="",
                )
            )
            continue

        head_dim = cfg["hidden_dim"] // cfg["num_heads"]
        engine = onnx_dir / f"trial_{tid:03d}.engine"
        log_path = log_dir / f"trial_{tid:03d}.trtexec.log"
        cmd = trtexec_cmd(
            image=args.image,
            sim_onnx=sim_path,
            engine=engine,
            hidden_dim=cfg["hidden_dim"],
            heads=cfg["num_heads"],
            head_dim=head_dim,
        )
        rc = run_cmd(cmd, cwd=out_root, log_file=log_path)

        if contains(log_path, "MyelinCheckException: tensor.cpp:852"):
            status = "myelin_fail_852"
        elif contains(log_path, "&&&& PASSED TensorRT.trtexec"):
            status = "pass"
        else:
            status = "other_fail" if rc != 0 else "unknown"

        results.append(
            TrialResult(
                trial_id=tid,
                cfg=cfg,
                onnx_path=str(onnx_path),
                sim_onnx_path=str(sim_path),
                node_count=node_count,
                trtexec_rc=rc,
                status=status,
                log_path=str(log_path),
            )
        )
        if status == "myelin_fail_852":
            break

    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    hit = [r for r in results if r.status == "myelin_fail_852"]
    print(f"[summary] trials={len(results)} hit_myelin_852={len(hit)} summary={summary_path}")
    if hit:
        h = hit[0]
        print("[hit]")
        print(f"  trial={h.trial_id} nodes={h.node_count}")
        print(f"  sim_onnx={h.sim_onnx_path}")
        print(f"  log={h.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
