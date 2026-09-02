#!/usr/bin/env python3
"""Build a tiny ONNX RoPE angle graph for INT64-vs-Gather experiments.

The graph deliberately contains only the position/angle path.  It is useful
for separating these questions from the full talker engine:

* ``math``: INT64 position -> FP32 Cast -> multiply by ``inv_freq`` -> cos/sin
* ``gather``: INT64 position -> Gather from a precomputed FP32 cos/sin table

Both variants expose FP32 and BF16 outputs.  TensorRT execution is intentionally
left to the caller (``trtexec`` or a TensorRT Python harness) so the generated
ONNX files can be inspected independently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


POSITIONS = np.asarray([255, 256, 257, 258, 511, 512, 513], dtype=np.int64)
DIM = 8
INV_FREQ = np.asarray(
    [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
    dtype=np.float32,
)


def _value_info(name: str, elem_type: int) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, [len(POSITIONS), DIM])


def _model(nodes: list[onnx.NodeProto], initializers: list[onnx.TensorProto]) -> onnx.ModelProto:
    positions = helper.make_tensor_value_info("position_ids", TensorProto.INT64, [len(POSITIONS)])
    graph = helper.make_graph(
        nodes,
        "rope_precision_micrograph",
        [positions],
        [
            _value_info("cos_fp32", TensorProto.FLOAT),
            _value_info("sin_fp32", TensorProto.FLOAT),
            _value_info("cos_bf16", TensorProto.BFLOAT16),
            _value_info("sin_bf16", TensorProto.BFLOAT16),
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="qwen3tts-rope-micrograph",
    )
    onnx.checker.check_model(model)
    return model


def build_math(path: Path) -> None:
    nodes = [
        helper.make_node("Cast", ["position_ids"], ["position_fp32"], name="position_cast", to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["position_fp32", "unsqueeze_axes"], ["position_column"], name="position_unsqueeze"),
        helper.make_node("Mul", ["position_column", "inv_freq"], ["angles"], name="angle_mul"),
        helper.make_node("Cos", ["angles"], ["cos_fp32"], name="cos"),
        helper.make_node("Sin", ["angles"], ["sin_fp32"], name="sin"),
        helper.make_node("Cast", ["cos_fp32"], ["cos_bf16"], name="cos_to_bf16", to=TensorProto.BFLOAT16),
        helper.make_node("Cast", ["sin_fp32"], ["sin_bf16"], name="sin_to_bf16", to=TensorProto.BFLOAT16),
    ]
    model = _model(
        nodes,
        [
            numpy_helper.from_array(INV_FREQ, name="inv_freq"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="unsqueeze_axes"),
        ],
    )
    onnx.save(model, path)


def build_gather(path: Path) -> None:
    table_pos = np.arange(1024, dtype=np.float32)[:, None]
    cos_table = np.cos(table_pos * INV_FREQ[None, :]).astype(np.float32)
    sin_table = np.sin(table_pos * INV_FREQ[None, :]).astype(np.float32)
    nodes = [
        helper.make_node("Gather", ["cos_table", "position_ids"], ["cos_fp32"], name="cos_gather", axis=0),
        helper.make_node("Gather", ["sin_table", "position_ids"], ["sin_fp32"], name="sin_gather", axis=0),
        helper.make_node("Cast", ["cos_fp32"], ["cos_bf16"], name="cos_to_bf16", to=TensorProto.BFLOAT16),
        helper.make_node("Cast", ["sin_fp32"], ["sin_bf16"], name="sin_to_bf16", to=TensorProto.BFLOAT16),
    ]
    model = _model(
        nodes,
        [
            numpy_helper.from_array(cos_table, name="cos_table"),
            numpy_helper.from_array(sin_table, name="sin_table"),
        ],
    )
    onnx.save(model, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/rope_precision_micrograph"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    build_math(args.out_dir / "rope_math.onnx")
    build_gather(args.out_dir / "rope_gather.onnx")
    np.save(args.out_dir / "positions.npy", POSITIONS)
    print(f"wrote {args.out_dir / 'rope_math.onnx'}")
    print(f"wrote {args.out_dir / 'rope_gather.onnx'}")
    print(f"positions={POSITIONS.tolist()}")


if __name__ == "__main__":
    main()
