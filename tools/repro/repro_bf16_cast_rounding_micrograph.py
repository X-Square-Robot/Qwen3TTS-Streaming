#!/usr/bin/env python3
"""Create a tiny FP32->BF16 Cast ONNX graph with rounding tie cases.

The graph is intentionally independent of the TTS model.  It is used with
TensorRT/trtexec to check whether a TensorRT BF16 reformat agrees with
PyTorch/CUDA's IEEE round-to-nearest-even conversion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


VALUES = np.asarray(
    [
        1.0,
        np.nextafter(np.float32(1.00390625), np.float32(0.0)),
        np.float32(1.00390625),
        np.nextafter(np.float32(1.00390625), np.float32(2.0)),
        np.float32(1.0078125),
        0.5,
        np.nextafter(np.float32(0.501953125), np.float32(0.0)),
        np.float32(0.501953125),
        np.nextafter(np.float32(0.501953125), np.float32(1.0)),
        -1.0,
        np.nextafter(np.float32(-1.00390625), np.float32(-2.0)),
        np.float32(-1.00390625),
        np.nextafter(np.float32(-1.00390625), np.float32(0.0)),
        255.0,
        256.0,
        257.0,
    ],
    dtype=np.float32,
)


def build(path: Path) -> None:
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [len(VALUES)])
    out = helper.make_tensor_value_info("y", TensorProto.BFLOAT16, [len(VALUES)])
    node = helper.make_node(
        "Cast", ["x"], ["y"], name="fp32_to_bf16_cast", to=TensorProto.BFLOAT16
    )
    graph = helper.make_graph([node], "bf16_cast_rounding", [inp], [out])
    model = helper.make_model(
        graph,
        producer_name="qwen3tts-repro",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    np.save(path.with_suffix(".input.npy"), VALUES)
    VALUES.tofile(path.with_suffix(".input.bin"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=Path, default=Path("/tmp/bf16_cast_rounding_micrograph")
    )
    args = parser.parse_args()
    build(args.out_dir / "cast.onnx")
    print("graph=", args.out_dir / "cast.onnx")
    print("values=", VALUES.tolist())
    expected = VALUES.astype(np.float32).view(np.uint32)
    # The high 16 bits after round-to-nearest-even are what PyTorch/CUDA BF16
    # conversion produces.  Keep this printout easy to compare with TRT.
    rounded = ((expected + 0x7FFF + ((expected >> 16) & 1)) >> 16).astype(np.uint16)
    print("expected_bf16_hex=", [hex(int(v)) for v in rounded])


if __name__ == "__main__":
    main()
