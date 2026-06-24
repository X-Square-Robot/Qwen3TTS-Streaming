"""Interactive TUI mode for qwen3tts CLI.

Reproduces the interactive experience of autorun.sh so that users can
just type ``qwen3tts`` and get a guided menu — no need to memorize
subcommands or flags.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from qwen3tts_tools.common import REPO_ROOT

# ── Colour helpers ──────────────────────────────────────────────────────

_CLR_RESET = "\033[0m"
_CLR_RED = "\033[31m"
_CLR_GREEN = "\033[32m"
_CLR_YELLOW = "\033[33m"
_CLR_BLUE = "\033[34m"
_CLR_BOLD = "\033[1m"

_USE_COLOUR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_CLR_RESET}" if _USE_COLOUR else text


def _info(msg: str) -> None:
    print(f"  {_c(_CLR_BLUE, 'ℹ')} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_c(_CLR_YELLOW, '⚠')} {msg}")


def _error(msg: str) -> None:
    print(f"  {_c(_CLR_RED, '✗')} {msg}", file=sys.stderr)


def _success(msg: str) -> None:
    print(f"  {_c(_CLR_GREEN, '✔')} {msg}")


# ── Input helpers ───────────────────────────────────────────────────────

def _is_interactive() -> bool:
    return sys.stdin.isatty()


def _prompt_with_default(prompt: str, default: str, timeout: int = 30) -> str:
    """Prompt the user with a default value; return default on timeout."""
    if not _is_interactive():
        return default
    try:
        raw = input(f"  {prompt} [{default}] ({timeout}s 后自动): ")
        return raw.strip() if raw.strip() else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def _prompt_choice(prompt: str, choices: list[str], default: str = "") -> str:
    """Prompt the user to pick from a list of choices."""
    if not _is_interactive():
        return default or choices[0]
    try:
        raw = input(f"  {prompt} [{default or choices[0]}]: ")
        val = raw.strip()
        if val in choices:
            return val
        return default or choices[0]
    except (EOFError, KeyboardInterrupt):
        print()
        return default or choices[0]


def _confirm(prompt: str, default: bool = False) -> bool:
    yes_no = "Y/n" if default else "y/N"
    if not _is_interactive():
        return default
    try:
        raw = input(f"  {prompt} [{yes_no}] (30s 后选 {'yes' if default else 'no'}): ")
        return raw.strip().lower() in ("y", "yes") if raw.strip() else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


# ── Banner ──────────────────────────────────────────────────────────────

def show_run_banner(phase_name: str, description: str, **extra: str | bool | None) -> None:
    """Print a phase-start banner (mirrors autorun.sh show_run_banner)."""
    print()
    print(_c(_CLR_BLUE, "╔══════════════════════════════════════════════════════════╗"))
    print(_c(_CLR_BLUE, f"║     Qwen3-TTS Triton — {phase_name}"))
    print(_c(_CLR_BLUE, "╚══════════════════════════════════════════════════════════╝"))
    print()
    print(f"  模式:      {description}")
    for key, value in extra.items():
        if value is None or value is False or value == "":
            continue
        label = key.replace("_", " ")
        if value is True:
            print(f"  {label}:    是")
        else:
            print(f"  {label}:    {value}")
    print()


# ── Variant auto-detection ──────────────────────────────────────────────

_KNOWN_VARIANTS = ["base-1.7b", "custom-1.7b", "design-1.7b", "base-0.6b", "custom-0.6b", "all-1.7b"]


def discover_exported_variants() -> list[str]:
    """List variants that have been exported to workspace/exported/."""
    exported_dir = REPO_ROOT / "workspace" / "exported"
    variants: list[str] = []
    if not exported_dir.is_dir():
        return variants
    for d in sorted(exported_dir.iterdir()):
        if d.is_dir() and not d.name.startswith(".") and (d / "weights").is_dir():
            variants.append(d.name)
    return variants


def select_model_variant(variant: str = "") -> str:
    """Interactively ask the user to pick a model variant."""
    if variant:
        return variant
    exported = discover_exported_variants()
    if exported:
        print()
        print("  可用的已导出变体:")
        for i, v in enumerate(exported, 1):
            print(f"    [{i}] {v}")
        if _is_interactive():
            choice = _prompt_with_default("请选择变体", exported[0] if exported else "custom-1.7b")
            return choice
        return exported[0]
    print()
    print("  已知变体:")
    for i, v in enumerate(_KNOWN_VARIANTS, 1):
        print(f"    [{i}] {v}")
    if _is_interactive():
        choice = _prompt_with_default("请选择变体", "custom-1.7b")
        return choice
    return "custom-1.7b"


# ── Resume-point detection ──────────────────────────────────────────────

def detect_resume_point() -> str:
    """Determine the furthest completed phase in the pipeline.

    Returns one of: "setup", "build", "package", "deploy", "done".
    """
    exported_dir = REPO_ROOT / "workspace" / "exported"
    model_repo_dir = REPO_ROOT / "workspace" / "model_repository"

    # Phase A complete = at least one variant with weights
    if not exported_dir.is_dir():
        return "setup"
    variants = discover_exported_variants()
    if not variants:
        return "setup"

    # Phase B complete = at least one variant has .engine files
    has_engines = False
    for v in variants:
        variant_dir = exported_dir / v
        for f in variant_dir.iterdir():
            if f.suffix == ".engine":
                has_engines = True
                break
        if has_engines:
            break
    if not has_engines:
        return "build"

    # Phase C1 complete = model_repository assembled
    if not model_repo_dir.is_dir() or not any(model_repo_dir.iterdir()):
        return "package"

    # Phase C2 complete = service running
    try:
        result = subprocess.run(
            ["pgrep", "-f", "engine.server"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "done"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=qwen3", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "done"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return "deploy"


# ── NGC tag selection ───────────────────────────────────────────────────

def select_ngc_tag_interactive(target_driver: str = "") -> str:
    """Interactively select an NGC container tag."""
    try:
        from qwen3tts_tools.ngc_matrix import (
            format_matrix_table,
            load_ngc_matrix,
            resolve_ngc_tag,
        )
        from qwen3tts_tools.docker import detect_driver_version
    except ImportError:
        _warn("ngc_matrix 模块不可用，跳过 NGC tag 选择")
        return ""

    driver = target_driver or detect_driver_version() or ""
    matrix = load_ngc_matrix()

    if driver:
        default_tag = resolve_ngc_tag(driver, matrix) or ""
        if default_tag:
            _info(f"当前驱动 {driver} 推荐的 NGC tag: {default_tag}")

    if not _is_interactive():
        return default_tag if driver else ""

    print()
    print("  可选 NGC 容器版本:")
    print(format_matrix_table(matrix, driver or None))
    print()
    choice = _prompt_with_default("选择 NGC tag (留空使用推荐)", default_tag or "")
    return choice


# ── GPU info ────────────────────────────────────────────────────────────

def show_gpu_info() -> None:
    """Print GPU status from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                print(f"    {line}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


# ── Cross-host interactive guide ────────────────────────────────────────

def interactive_cross_host_guide() -> None:
    """Guide the user through cross-host engine compilation."""
    print()
    print("  跨机 Engine 编译引导")
    print()
    print("  适用场景：当前机器负责导图/打包，但 TensorRT engine 要在生产同构 GPU 上编译。")
    print()
    print("  [0] discover-target — 获取目标机器 target_profile.json")
    print("      支持本地 / SSH 远端 / 复制粘贴 三种方式")
    print("  [1] 在当前机器采集 target_profile.json (= [0] --local)")
    print("  [2] 生成离线 engine_build_bundle.tar.zst")
    print("  [3] 导入目标机器返回的 engine_artifact_bundle.tar.zst")
    print("  [4] SSH 远端编译并自动导入")
    print("  [5] 打印完整流程命令")
    print("  [q] 返回/退出")
    print()

    choice = _prompt_choice("请选择 [0-5/q]", ["0", "1", "2", "3", "4", "5", "q", "Q"], default="5")

    if choice in ("q", "Q"):
        print("  已退出跨机引导。")
        return

    if choice == "0":
        _guide_discover_target()
    elif choice == "1":
        out = _prompt_with_default(
            "target_profile.json 输出路径",
            str(REPO_ROOT / "workspace" / "target_profile.json"),
        )
        _run_cli(["probe", "--output", out])
    elif choice == "2":
        variant = select_model_variant()
        tp = _prompt_with_default(
            "目标机器 target_profile.json 路径",
            str(REPO_ROOT / "workspace" / "target_profile.json"),
        )
        out = _prompt_with_default(
            "engine build bundle 输出路径",
            str(REPO_ROOT / "workspace" / "engine_build_bundle.tar.zst"),
        )
        dtype, io_dtype = _prompt_build_dtypes()
        _run_cli(["build", "make-bundle", "-m", variant, "--target-profile", tp, "--out", out, "--dtype", dtype])
    elif choice == "3":
        artifact = _prompt_with_default(
            "engine artifact bundle 路径",
            str(REPO_ROOT / "workspace" / "engine_artifact_bundle.tar.zst"),
        )
        _run_cli(["build", "import-artifact", artifact])
    elif choice == "4":
        variant = select_model_variant()
        tp = _prompt_with_default(
            "目标机器 target_profile.json 路径",
            str(REPO_ROOT / "workspace" / "target_profile.json"),
        )
        remote_host = _prompt_with_default("SSH 目标主机 user@host", "user@prod-gpu-host")
        remote_workdir = _prompt_with_default("远端工作目录", "/tmp/qwen3-tts-engine-build")
        dtype, io_dtype = _prompt_build_dtypes()
        _run_cli([
            "build", "remote-build", "-m", variant,
            "--target-profile", tp,
            "--remote-host", remote_host,
            "--remote-workdir", remote_workdir,
            "--dtype", dtype,
        ])
    elif choice == "5":
        _print_cross_host_help()


def _guide_discover_target() -> None:
    """Sub-guide for discover-target mode selection."""
    print()
    print("  选择获取方式:")
    print("    [a] --local    当前主机就是生产目标")
    print("    [b] --remote   SSH 到生产机器")
    print("    [c] --paste    打印探测脚本并粘贴回 JSON")
    mode = _prompt_choice("请选择 [a/b/c]", ["a", "b", "c"], default="a")

    out = _prompt_with_default(
        "target_profile.json 输出路径",
        str(REPO_ROOT / "workspace" / "target_profile.json"),
    )

    if mode == "a":
        _run_cli(["discover-target", "--local", "--output", out])
    elif mode == "b":
        remote_host = _prompt_with_default("SSH 目标主机 user@host", "user@prod-gpu-host")
        _run_cli(["discover-target", "--remote-host", remote_host, "--output", out])
    elif mode == "c":
        _run_cli(["discover-target", "--paste", "--output", out])


def _prompt_build_dtypes() -> tuple[str, str]:
    """Prompt for engine dtype and triton IO float dtype."""
    print()
    print("  精度配置 (阶段 B)")
    print("    引擎精度: TensorRT builder/compute precision")
    print("    I/O 精度: fused TensorRT/Triton float binding dtype，默认跟随引擎精度")

    engine_dtype = _prompt_choice("引擎精度 bf16|fp16|fp32|fp8", ["bf16", "fp16", "fp32", "fp8"], default="bf16")
    io_default = engine_dtype if engine_dtype in ("bf16", "fp16", "fp32") else "fp32"
    io_dtype = _prompt_choice("浮点 I/O 精度 bf16|fp16|fp32", ["bf16", "fp16", "fp32"], default=io_default)
    return engine_dtype, io_dtype


def _print_cross_host_help() -> None:
    """Print the complete cross-host workflow instructions."""
    print("""
  离线推荐流程：

    # 1. 在生产同构 GPU 机器上：
    qwen3tts probe --output target_profile.json

    # 2. 把 target_profile.json 拷回当前机器，生成构建包：
    qwen3tts build make-bundle -m custom-1.7b \\
      --target-profile target_profile.json \\
      --dtype bf16 \\
      --out workspace/engine_build_bundle.tar.zst

    # 3. 把 engine_build_bundle.tar.zst 拷到目标机器并编译：
    mkdir -p /tmp/qwen3-engine-build
    tar --zstd -xf engine_build_bundle.tar.zst -C /tmp/qwen3-engine-build
    cd /tmp/qwen3-engine-build
    bash build_on_target.sh

    # 4. 把 engine_artifact_bundle.tar.zst 拷回当前机器并导入：
    qwen3tts build import-artifact workspace/engine_artifact_bundle.tar.zst

    # 5. 组装模型包并构建 engine 镜像：
    qwen3tts package -m custom-1.7b --gateway engine-docker --build

    # 6. 启动服务：
    qwen3tts run -m custom-1.7b --gateway engine-docker

  SSH 自动流程：

    qwen3tts build remote-build -m custom-1.7b \\
      --target-profile target_profile.json \\
      --dtype bf16 \\
      --remote-host user@prod-gpu-host \\
      --remote-workdir /tmp/qwen3-engine-build
""")


def _run_cli(args: list[str]) -> int:
    """Run a qwen3tts CLI subcommand in-process."""
    from qwen3tts_cli.main import main
    try:
        return main(args)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


# ── Main interactive mode ───────────────────────────────────────────────

def interactive_mode() -> int:
    """Launch the interactive TUI — the experience users get from bare ``qwen3tts``."""
    # Show current status
    try:
        from qwen3tts_tools.status import check_all, format_status
        status = check_all()
        print(format_status(status))
    except ImportError:
        pass

    # Determine resume point
    resume_point = detect_resume_point()
    resume_labels = {"setup": "setup", "build": "build", "package": "package", "deploy": "run", "done": "done"}
    resume_label = resume_labels.get(resume_point, resume_point)

    print("  要执行什么操作？")
    print()
    print("  [1] 完整本机流程  (setup → build → package → run)")
    print("  [2] 环境配置      (阶段 A)")
    print("  [3] 构建引擎      (阶段 B)")
    print("  [4] 组装/打包产物 (不启动服务)")
    print("  [5] 当前机器启动服务 (run，不做跨机部署)")
    if resume_point != "done":
        print(f"  [6] 从 {resume_label} 恢复")
    else:
        print("  [6] 已完成/查看当前状态")
    print("  [7] 查看详细状态")
    print("  [8] 跨机编译引导")
    print("  [q] 退出")
    print()

    choice = _prompt_choice("请选择 [1-8/q]", ["1", "2", "3", "4", "5", "6", "7", "8", "q", "Q"], default="1")

    if choice in ("q", "Q"):
        print("  已退出。")
        return 0

    if choice == "7":
        try:
            from qwen3tts_tools.status import check_all, format_status
            status = check_all()
            print(format_status(status))
        except ImportError:
            _warn("status 模块不可用")
        return 0

    if choice == "8":
        interactive_cross_host_guide()
        return 0

    # ── Collect parameters for phases ──

    # Variant
    variant = ""
    needs_variant = choice in ("1", "2", "3", "4", "5", "6")
    if needs_variant:
        print()
        variant = select_model_variant()

    # Build flags (needed for Phase B)
    needs_build = choice in ("1", "3") or (choice == "6" and resume_point in ("setup", "build"))
    engine_dtype = "bf16"
    triton_io_dtype = "bf16"
    ngc_tag = ""
    max_batch_size = ""
    max_input_len = ""
    max_seq_len = ""

    if needs_build:
        # Engine dtype
        print()
        engine_dtype, triton_io_dtype = _prompt_build_dtypes()

        # NGC tag
        print()
        print("  NGC 容器版本")
        ngc_tag = select_ngc_tag_interactive()

        # Build profile
        print()
        print("  TensorRT 构建 profile (留空则 Phase B 按构建 GPU 显存给建议默认值)")
        max_batch_size = _prompt_with_default("max-batch-size [auto]", "auto")
        max_input_len = _prompt_with_default("max-input-len [auto]", "auto")
        max_seq_len = _prompt_with_default("max-seq-len [auto]", "auto")
        if max_batch_size == "auto":
            max_batch_size = ""
        if max_input_len == "auto":
            max_input_len = ""
        if max_seq_len == "auto":
            max_seq_len = ""

    # Phase C flags
    will_package = choice in ("1", "4") or (choice == "6" and resume_point in ("setup", "build", "package"))
    will_deploy = choice in ("1", "5") or (choice == "6" and resume_point != "done")
    gateway = "standalone"

    if will_package or will_deploy:
        # GPU selection
        if not _is_interactive() or not shutil.which("nvidia-smi"):
            gpu = ""
        else:
            print()
            print("  GPU 选择 (默认: auto，自动选择当前空闲显存最多的 GPU)")
            show_gpu_info()
            gpu = _prompt_with_default("GPU [auto]", "auto")
            if gpu == "auto":
                gpu = ""

        # Gateway mode
        print()
        print("  阶段 C — 产物/运行方式:")
        print("    [1] standalone    — 组装模型包；run 时本机 Python 运行 engine")
        print("    [2] triton        — 组装 model_repository；package 可构建自包含 Triton 镜像")
        print("    [3] engine-docker — 组装模型包并构建 engine 镜像；run 时挂载模型包启动容器")
        print()
        gw_choice = _prompt_choice("请选择 [1-3]", ["1", "2", "3"], default="1")
        gateway = {"1": "standalone", "2": "triton", "3": "engine-docker"}.get(gw_choice, "standalone")
        _info(f"已选择 Phase C 方式: {gateway}")

        # Runtime profile (only for deploy)
        runtime_max_batch = ""
        runtime_max_seq_len = ""
        if will_deploy:
            print()
            print("  Runtime 上限 (留空则读取 manifest engine_profile；不能超过 Phase B profile)")
            runtime_max_batch = _prompt_with_default("runtime-max-batch-size [manifest]", "")
            runtime_max_seq_len = _prompt_with_default("runtime-max-seq-len [manifest]", "")

    # ── Build CLI args and dispatch ──

    def _build_cli_args() -> list[str]:
        """Assemble CLI arguments based on collected parameters."""
        args: list[str] = []
        if variant:
            args += ["-m", variant]
        if engine_dtype and engine_dtype != "bf16":
            args += ["--dtype", engine_dtype]
        if triton_io_dtype and triton_io_dtype != engine_dtype:
            args += ["--triton-io-float-dtype", triton_io_dtype]
        if ngc_tag:
            args += ["--ngc-tag", ngc_tag]
        if max_batch_size:
            args += ["--max-batch-size", max_batch_size]
        if max_input_len:
            args += ["--max-input-len", max_input_len]
        if max_seq_len:
            args += ["--max-seq-len", max_seq_len]
        if gateway != "standalone":
            args += ["--gateway", gateway]
        return args

    if choice == "1":
        show_run_banner("完整本机流程", "A → B → package → run", 变体=variant or None, 引擎精度=engine_dtype)
        cli_args = ["all"] + _build_cli_args()
        result = _run_cli(cli_args)
        if result == 0:
            print()
            _success("全部阶段完成！部署产物已组装，TTS 服务已在当前机器运行。")
        return result

    elif choice == "2":
        show_run_banner("阶段 A", "环境 + 导出", 变体=variant or None)
        return _run_cli(["setup"] + _build_cli_args())

    elif choice == "3":
        show_run_banner("阶段 B", "TensorRT 引擎", 变体=variant or None, 引擎精度=engine_dtype)
        return _run_cli(["build"] + _build_cli_args())

    elif choice == "4":
        show_run_banner("阶段 C", "组装/打包产物（不启动服务）", 变体=variant or None, 阶段C=gateway)
        return _run_cli(["package"] + _build_cli_args())

    elif choice == "5":
        show_run_banner("阶段 C", "当前机器启动 TTS 服务", 变体=variant or None, 阶段C=gateway)
        return _run_cli(["run"] + _build_cli_args())

    elif choice == "6":
        # Resume from the detected point
        if resume_point == "setup":
            show_run_banner("恢复流程", f"从 {resume_label} 继续", 变体=variant or None)
            return _run_cli(["all"] + _build_cli_args())
        elif resume_point == "build":
            show_run_banner("恢复流程", f"从 {resume_label} 继续", 变体=variant or None)
            _run_cli(["build"] + _build_cli_args())
            print()
            _run_cli(["package"] + _build_cli_args())
            print()
            return _run_cli(["run"] + _build_cli_args())
        elif resume_point == "package":
            show_run_banner("恢复流程", f"从 {resume_label} 继续", 变体=variant or None)
            _run_cli(["package"] + _build_cli_args())
            print()
            return _run_cli(["run"] + _build_cli_args())
        elif resume_point == "deploy":
            show_run_banner("恢复流程", f"从 {resume_label} 继续", 变体=variant or None)
            return _run_cli(["run"] + _build_cli_args())
        else:  # done
            return _run_cli(["status"])

    return 0
