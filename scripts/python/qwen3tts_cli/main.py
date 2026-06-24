"""Qwen3-TTS CLI main entry point with argparse subcommands.

When invoked without a subcommand, launches the interactive TUI mode
that mirrors the autorun.sh guided experience.
"""

from __future__ import annotations

import argparse
import sys


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across most subcommands."""
    parser.add_argument(
        "-m", "--variant", default="",
        help="Model variant (default: auto-discover)",
    )
    parser.add_argument(
        "--model-version", type=int, default=1,
        help="Triton model version directory (default: 1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--ngc-tag", default="",
        help="Select NGC tritonserver tag (e.g. 25.03)",
    )
    parser.add_argument(
        "--target-driver", default="",
        help="Target NVIDIA driver version for NGC container selection",
    )
    parser.add_argument(
        "--target-profile", default="",
        help="Path to target_profile.json for cross-host builds",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code (0 = success)."""
    parser = argparse.ArgumentParser(
        prog="qwen3tts",
        description="Qwen3-TTS Triton lifecycle CLI — Python native.\n\n"
                    "Run without arguments for interactive (TUI) mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmations (non-interactive)",
    )
    sub = parser.add_subparsers(dest="command")

    # ── Phase pipeline ──
    p_all = sub.add_parser("all", help="Full pipeline: setup → build → package → deploy")
    _add_common_args(p_all)
    p_all.add_argument("--source", default="auto", choices=["auto", "hf", "modelscope"],
                       help="Model download source (default: auto)")
    p_all.add_argument("--skip-download", action="store_true", help="Skip model download")
    p_all.add_argument("--skip-export", action="store_true", help="Skip ONNX export")
    p_all.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    p_all.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "fp8"],
                       help="Engine dtype (default: bf16)")
    p_all.add_argument("--triton-io-float-dtype", default="",
                       help="Triton IO float dtype (default: same as --dtype)")
    p_all.add_argument("--backbone-precision", default="",
                       choices=["fp32", "bf16", "fp16", "fp8"],
                       help="Backbone compute precision (default: same as --dtype)")
    p_all.add_argument("--cp-precision", default="",
                       choices=["fp32", "bf16", "fp16", "fp8"],
                       help="Code Predictor compute precision (default: same as --dtype; "
                            "set to fp32 to mitigate BF16 numerical sensitivity)")
    p_all.add_argument("--code2wav-precision", default="",
                       choices=["fp32", "bf16", "fp16", "fp8"],
                       help="Code2Wav compute precision (default: same as --dtype)")
    p_all.add_argument("--max-batch-size", type=int, default=0, help="Max batch size for TRT profile")
    p_all.add_argument("--max-input-len", type=int, default=0, help="Max input length for prefill")
    p_all.add_argument("--max-seq-len", type=int, default=0, help="Max sequence length for TRT profile")
    p_all.add_argument("--device", default="auto", help="GPU device for all phases (auto|N|cuda:N)")
    p_all.add_argument("--export-device", default="", help="Override Phase A export GPU only")
    p_all.add_argument("--build-device", default="", help="Override Phase B build GPU only")
    p_all.add_argument("--runtime-device", default="", help="Override Phase C serving GPU only")
    p_all.add_argument("--gateway", default="standalone",
                       choices=["standalone", "triton", "engine-docker"],
                       help="Gateway mode (default: standalone)")
    p_all.add_argument("--engine-mode", default="trt", choices=["onnx", "trt"],
                       help="Engine mode (default: trt)")
    p_all.add_argument("--image", default="", help="Override NGC container image for trtexec")
    p_all.add_argument("--engine-image", default="", help="Engine-docker image tag")
    p_all.add_argument("--port", type=int, default=50051, help="Standalone/engine gRPC port")
    p_all.add_argument("--grpc-port", type=int, default=8001, help="Triton gRPC port")
    p_all.add_argument("--http-port", type=int, default=8000, help="Triton HTTP port")
    p_all.add_argument("--build", "--rebuild-image", action="store_true",
                       help="Rebuild runtime image before package/deploy")
    p_all.add_argument("rest", nargs=argparse.REMAINDER, help="Extra args forwarded to phases")

    # ── Phase A: setup ──
    p_setup = sub.add_parser("setup", help="Phase A: install environment, download/export models")
    _add_common_args(p_setup)
    p_setup.add_argument("--source", default="auto", choices=["auto", "hf", "modelscope"],
                         help="Model download source (default: auto)")
    p_setup.add_argument("--skip-download", action="store_true", help="Skip model download")
    p_setup.add_argument("--skip-export", action="store_true", help="Skip ONNX export")
    p_setup.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    p_setup.add_argument("--python", default="", help="Python version (default: from NGC matrix)")
    p_setup.add_argument("--env-name", default="", help="Virtual env name (default: qwen3-tts)")
    p_setup.add_argument("--export-device", default="", help="Override Phase A export GPU only")

    # ── Phase B: build ──
    p_build = sub.add_parser("build", help="Phase B: compile TensorRT engines")
    _add_common_args(p_build)
    p_build.add_argument("--max-batch-size", type=int, default=0, help="Max batch size for TRT profile")
    p_build.add_argument("--max-input-len", type=int, default=0, help="Max input length for prefill")
    p_build.add_argument("--max-seq-len", type=int, default=0, help="Max sequence length for TRT profile")
    p_build.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "fp8"],
                         help="Engine dtype (default: bf16)")
    p_build.add_argument("--triton-io-float-dtype", default="",
                         help="Triton IO float dtype (default: same as --dtype)")
    p_build.add_argument("--backbone-precision", default="",
                         choices=["fp32", "bf16", "fp16", "fp8"],
                         help="Backbone compute precision (default: same as --dtype)")
    p_build.add_argument("--cp-precision", default="",
                         choices=["fp32", "bf16", "fp16", "fp8"],
                         help="Code Predictor compute precision (default: same as --dtype; "
                              "set to fp32 to mitigate BF16 numerical sensitivity)")
    p_build.add_argument("--code2wav-precision", default="",
                         choices=["fp32", "bf16", "fp16", "fp8"],
                         help="Code2Wav compute precision (default: same as --dtype)")
    p_build.add_argument("--image", default="", help="Override NGC container image for trtexec")
    p_build.add_argument("--device", default="auto", help="GPU device (auto|N|cuda:N)")
    p_build.add_argument("--build-device", default="", help="Override Phase B build GPU only")
    p_build.add_argument("--out", default="workspace/engine_build_bundle.tar.zst",
                         help="Output path for build bundle")
    p_build.add_argument("--allow-fingerprint-mismatch", action="store_true",
                         help="Skip fingerprint validation on import")
    p_build.add_argument("--pull-only", action="store_true",
                         help="Pull the NGC container image and exit")

    # Build sub-subcommands
    build_sub = p_build.add_subparsers(dest="build_action")

    build_sub.add_parser("local", help="Local engine compilation (default)")

    p_mb = build_sub.add_parser("make-bundle", help="Create cross-host build bundle")
    p_mb.add_argument("--target-profile", required=True,
                       help="Path to target_profile.json (required)")
    p_mb.add_argument("--out", default="workspace/engine_build_bundle.tar.zst",
                       help="Output bundle path")

    p_ia = build_sub.add_parser("import-artifact", help="Import compiled engines from artifact bundle")
    p_ia.add_argument("artifact", help="Path to engine_artifact_bundle.tar.zst")
    p_ia.add_argument("--allow-fingerprint-mismatch", action="store_true",
                       help="Skip fingerprint validation")

    p_rb = build_sub.add_parser("remote-build", help="SSH-driven remote engine compilation")
    p_rb.add_argument("--target-profile", required=True,
                       help="Path to target_profile.json (required)")
    p_rb.add_argument("--remote-host", required=True,
                       help="SSH host (e.g. user@prod-gpu-host)")
    p_rb.add_argument("--remote-workdir", default="/tmp/qwen3-tts-engine-build",
                       help="Remote working directory")

    # ── Phase C1: package ──
    p_package = sub.add_parser("package", help="Phase C1: assemble deployment artifacts")
    _add_common_args(p_package)
    p_package.add_argument("--gateway", default="standalone",
                           choices=["standalone", "triton", "engine-docker"],
                           help="Gateway mode (default: standalone)")
    p_package.add_argument("--engine-mode", default="trt", choices=["onnx", "trt"],
                           help="Engine mode for model repo assembly (default: trt)")
    p_package.add_argument("--build", "--rebuild-image", action="store_true",
                           help="Rebuild runtime image before packaging")
    p_package.add_argument("--engine-image", default="", help="Engine-docker image tag")
    p_package.add_argument("--image", default="", help="Override NGC container image")

    # ── Phase C2: deploy / run ──
    p_run = sub.add_parser("run", help="Phase C2: start the TTS service")
    _add_common_args(p_run)
    p_run.add_argument("--gateway", default="standalone",
                       choices=["standalone", "triton", "engine-docker"],
                       help="Gateway mode (default: standalone)")
    p_run.add_argument("--engine-mode", default="trt", choices=["onnx", "trt"],
                       help="Engine mode (default: trt)")
    p_run.add_argument("--port", type=int, default=50051, help="gRPC port (default: 50051)")
    p_run.add_argument("--ws-port", type=int, default=50052, help="WebSocket port (default: 50052)")
    p_run.add_argument("--device", default="auto", help="GPU device (auto|N|cuda:N)")
    p_run.add_argument("--runtime-device", default="", help="Override Phase C serving GPU only")
    p_run.add_argument("--max-batch", type=int, default=0, help="Max batch size")
    p_run.add_argument("--max-sessions", type=int, default=128, help="Max concurrent sessions (default: 128)")
    p_run.add_argument("--max-seq-len", type=int, default=0, help="Max sequence length for runtime")
    p_run.add_argument("--foreground", action="store_true", help="Run in foreground")
    p_run.add_argument("--image", default="", help="Override container image")
    p_run.add_argument("--engine-image", default="", help="Engine-docker image tag")
    p_run.add_argument("--build", "--rebuild-image", action="store_true",
                       help="Rebuild runtime image before deploy")

    # ── stop ──
    p_stop = sub.add_parser("stop", help="Stop the TTS service")
    p_stop.add_argument("--variant", "-m", default="", help="Model variant")

    # ── status ──
    p_status = sub.add_parser("status", help="Show current pipeline status")
    p_status.add_argument("--json", action="store_true", help="Output as JSON")

    # ── probe ──
    p_probe = sub.add_parser("probe", help="Probe target GPU/driver profile")
    p_probe.add_argument("--output", default="workspace/target_profile.json",
                         help="Output path for profile JSON")

    # ── discover-target ──
    p_discover = sub.add_parser("discover-target", help="Acquire target_profile.json from production target")
    p_discover.add_argument("--output", default="workspace/target_profile.json",
                            help="Output path for profile JSON")
    p_discover.add_argument("--local", action="store_const", const="local", dest="discover_mode",
                            help="Run probe on the current host")
    p_discover.add_argument("--remote-host", default="",
                            help="SSH host for remote discovery (e.g. user@prod-gpu-host)")
    p_discover.add_argument("--remote-workdir", default="/tmp/qwen3-tts-engine-build",
                            help="Remote working directory")
    p_discover.add_argument("--paste", action="store_const", const="paste", dest="discover_mode",
                            help="Print probe script and accept pasted JSON")

    # ── download ──
    p_download = sub.add_parser("download", help="Download model weights")
    _add_common_args(p_download)
    p_download.add_argument("--source", default="auto", choices=["auto", "hf", "modelscope"],
                           help="Download source (default: auto)")

    # ── assemble ──
    p_assemble = sub.add_parser("assemble", help="Assemble Triton model_repository")
    _add_common_args(p_assemble)
    p_assemble.add_argument("--engine-mode", default="trt", choices=["onnx", "trt"],
                            help="Engine mode (default: trt)")

    # ── pull ──
    p_pull = sub.add_parser("pull", help="Pull NGC Triton container image")

    # ── list-ngc ──
    p_list_ngc = sub.add_parser("list-ngc", help="List selectable NGC container versions")
    p_list_ngc.add_argument("--target-driver", default="", help="Target NVIDIA driver version")

    # ── update-matrix ──
    sub.add_parser("update-matrix", help="Update NGC compatibility matrix from NVIDIA website")

    # ── Parse and dispatch ──
    args = parser.parse_args(argv)
    if args.command is None:
        # No subcommand → launch interactive TUI mode
        from qwen3tts_cli.interactive import interactive_mode
        return interactive_mode()

    # Resolve discover-target mode from flags
    if args.command == "discover-target":
        if not getattr(args, "discover_mode", ""):
            if getattr(args, "remote_host", ""):
                args.discover_mode = "remote"
            else:
                args.discover_mode = "local"

    # Dispatch to subcommand modules
    dispatch = {
        "all": _cmd_all,
        "setup": _cmd_setup,
        "build": _cmd_build,
        "package": _cmd_package,
        "run": _cmd_run,
        "stop": _cmd_stop,
        "status": _cmd_status,
        "probe": _cmd_probe,
        "discover-target": _cmd_discover_target,
        "download": _cmd_download,
        "assemble": _cmd_assemble,
        "pull": _cmd_pull,
        "list-ngc": _cmd_list_ngc,
        "update-matrix": _cmd_update_matrix,
    }
    handler = dispatch.get(args.command)
    if handler:
        result = handler(args)
        return result if isinstance(result, int) else 0
    else:
        parser.print_help()
        return 1


# ── Subcommand handlers ──


def _cmd_all(args: argparse.Namespace) -> int:
    """Run full pipeline: setup → build → package → deploy."""
    from qwen3tts_cli.interactive import show_run_banner
    from qwen3tts_cli.cmd_setup import run_setup
    from qwen3tts_cli.cmd_build import run_build
    from qwen3tts_cli.cmd_deploy import run_package, run_deploy

    show_run_banner(
        "完整本机流程", "A → B → package → run",
        变体=getattr(args, "variant", "") or None,
        引擎精度=getattr(args, "dtype", "bf16"),
        阶段C=getattr(args, "gateway", "standalone"),
    )

    if run_setup(args) != 0:
        return 1
    print()
    if run_build(args) != 0:
        return 1
    print()
    if run_package(args) != 0:
        return 1
    print()
    if run_deploy(args) != 0:
        return 1

    print()
    if getattr(args, "dry_run", False):
        from qwen3tts_cli.interactive import _c, _CLR_GREEN
        print(_c(_CLR_GREEN, "预演完成：未实际组装产物或启动服务。"))
    else:
        from qwen3tts_cli.interactive import _c, _CLR_GREEN
        print(_c(_CLR_GREEN, "全部阶段完成！部署产物已组装，TTS 服务已在当前机器运行。"))
    return 0


def _cmd_setup(args: argparse.Namespace) -> int:
    from qwen3tts_cli.interactive import show_run_banner
    from qwen3tts_cli.cmd_setup import run_setup
    show_run_banner("阶段 A", "环境 + 导出", 变体=getattr(args, "variant", "") or None)
    return run_setup(args)


def _cmd_build(args: argparse.Namespace) -> int:
    from qwen3tts_cli.interactive import show_run_banner
    from qwen3tts_cli.cmd_build import run_build
    show_run_banner(
        "阶段 B", "TensorRT 引擎",
        变体=getattr(args, "variant", "") or None,
        引擎精度=getattr(args, "dtype", "bf16"),
    )
    return run_build(args)


def _cmd_package(args: argparse.Namespace) -> int:
    from qwen3tts_cli.interactive import show_run_banner
    from qwen3tts_cli.cmd_deploy import run_package
    show_run_banner(
        "阶段 C", "组装/打包产物（不启动服务）",
        变体=getattr(args, "variant", "") or None,
        阶段C=getattr(args, "gateway", "standalone"),
    )
    return run_package(args)


def _cmd_run(args: argparse.Namespace) -> int:
    from qwen3tts_cli.interactive import show_run_banner
    from qwen3tts_cli.cmd_deploy import run_deploy
    show_run_banner(
        "阶段 C", "当前机器启动 TTS 服务",
        变体=getattr(args, "variant", "") or None,
        阶段C=getattr(args, "gateway", "standalone"),
    )
    return run_deploy(args)


def _cmd_stop(args: argparse.Namespace) -> int:
    from qwen3tts_cli.cmd_deploy import run_stop
    return run_stop(args)


def _cmd_status(args: argparse.Namespace) -> int:
    from qwen3tts_cli.cmd_status import run_status
    return run_status(args)


def _cmd_probe(args: argparse.Namespace) -> int:
    from qwen3tts_cli.cmd_probe import run_probe
    return run_probe(args)


def _cmd_discover_target(args: argparse.Namespace) -> int:
    from qwen3tts_cli.cmd_discover import run_discover_target
    return run_discover_target(args)


def _cmd_download(args: argparse.Namespace) -> int:
    from qwen3tts_cli.cmd_download import run_download
    return run_download(args)


def _cmd_assemble(args: argparse.Namespace) -> int:
    from qwen3tts_cli.cmd_deploy import run_assemble
    return run_assemble(args)


def _cmd_pull(args: argparse.Namespace) -> int:
    from qwen3tts_cli.cmd_deploy import run_pull
    return run_pull(args)


def _cmd_list_ngc(args: argparse.Namespace) -> int:
    """List selectable NGC container versions."""
    try:
        from qwen3tts_tools.ngc_matrix import format_matrix_table, load_ngc_matrix
        from qwen3tts_tools.docker import detect_driver_version

        target_driver = getattr(args, "target_driver", "") or ""
        driver = target_driver or detect_driver_version() or None
        matrix = load_ngc_matrix()
        print(format_matrix_table(matrix, driver))
        return 0
    except ImportError:
        print("Error: ngc_matrix module not available.", file=sys.stderr)
        return 1


def _cmd_update_matrix(args: argparse.Namespace) -> int:
    """Update NGC compatibility matrix from NVIDIA website."""
    from qwen3tts_tools.common import REPO_ROOT

    bash_script = REPO_ROOT / "scripts" / "bash" / "lib" / "ngc_updater.sh"
    if not bash_script.is_file():
        print("Error: NGC updater script not found.", file=sys.stderr)
        return 1

    import subprocess
    return subprocess.call(["bash", str(bash_script)])


def cli_main() -> None:
    """Entry point registered in pyproject.toml — calls main() and sys.exit()."""
    sys.exit(main())
