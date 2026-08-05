from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_ENV_VARS = (
    "QWEN_LOG_DIR",
    "QWEN_LOG_MAX_BYTES",
    "QWEN_LOG_BACKUP_COUNT",
    "QWEN_LOG_STDOUT",
)


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    start_at = text.index(start)
    end_at = text.index(end, start_at + len(start))
    return text[start_at:end_at]


def _compose_service(text: str, service: str, next_service: str) -> str:
    return _between(text, f"  {service}:\n", f"  {next_service}:\n")


def _yaml_key_block(service_block: str, key: str) -> str:
    match = re.search(
        rf"(?ms)^    {re.escape(key)}:\s*.*?(?=^    [A-Za-z][A-Za-z0-9_-]*:|\Z)",
        service_block,
    )
    assert match is not None, f"missing {key!r} block"
    return match.group(0)


def _assert_runner_precedes(
    command_text: str,
    child_command: str,
    runner_reference: str = "rotating_log_runner.py",
) -> None:
    runner_at = command_text.find(runner_reference)
    child_at = command_text.find(child_command)
    assert runner_at >= 0, "command does not invoke the rotating log runner"
    assert child_at > runner_at, f"{child_command!r} is not wrapped by the log runner"
    assert "--service" in command_text
    assert "--" in command_text


def _assert_generated_dockerfile_wraps_triton(function_text: str) -> None:
    copy_lines = [
        line
        for line in function_text.splitlines()
        if line.lstrip().upper().startswith("COPY ")
    ]
    assert any("rotating_log_runner.py" in line for line in copy_lines)

    startup_lines = [
        line
        for line in function_text.splitlines()
        if line.lstrip().upper().startswith(("ENTRYPOINT ", "CMD "))
    ]
    _assert_runner_precedes("\n".join(startup_lines), "tritonserver")


@pytest.mark.parametrize(
    "dockerfile",
    (
        "infra/docker/Dockerfile.engine",
        "infra/docker/Dockerfile.triton",
    ),
)
def test_runtime_dockerfiles_copy_the_log_runner(dockerfile: str):
    text = _read(dockerfile)
    copy_lines = [
        line for line in text.splitlines() if line.lstrip().upper().startswith("COPY ")
    ]

    # Copying the complete compose helper directory also includes the runner.
    assert any(
        "rotating_log_runner.py" in line or "scripts/compose/" in line
        for line in copy_lines
    )


@pytest.mark.parametrize(
    ("service", "next_service", "child_command"),
    (
        ("engine", "triton", "engine-entrypoint.sh"),
        ("triton", "demo-api", "tritonserver"),
    ),
)
def test_compose_services_wrap_commands_and_expose_log_settings(
    service: str,
    next_service: str,
    child_command: str,
):
    service_block = _compose_service(
        _read("infra/docker/compose.yaml"), service, next_service
    )
    command_block = _yaml_key_block(service_block, "command")
    environment_block = _yaml_key_block(service_block, "environment")
    volumes_block = _yaml_key_block(service_block, "volumes")

    _assert_runner_precedes(command_block, child_command)
    assert "/tmp/qwen3tts_rotating_log_runner.py" in command_block
    assert re.search(rf"\b{re.escape(service)}\b", command_block)
    assert re.search(r"(?m)^    init:\s+true\s*$", service_block)
    assert "scripts/compose/rotating_log_runner.py" in volumes_block
    assert "/tmp/qwen3tts_rotating_log_runner.py:ro" in volumes_block

    for name in LOG_ENV_VARS:
        assert re.search(rf"(?m)^\s+{name}:\s*", environment_block)
        assert f"${{{name}" in environment_block


@pytest.mark.parametrize(
    (
        "script",
        "function_start",
        "function_end",
        "runner_variable",
        "service",
        "child_command",
    ),
    (
        (
            "scripts/bash/lib/engine.sh",
            "engine_start_docker() {",
            "engine_stop_docker() {",
            "ENGINE_CONTAINER_LOG_RUNNER",
            "engine",
            "python3 -m engine.server",
        ),
        (
            "scripts/bash/lib/triton.sh",
            "triton_run() {",
            "triton_health_check() {",
            "TRITON_CONTAINER_LOG_RUNNER",
            "triton",
            "tritonserver",
        ),
    ),
)
def test_direct_docker_run_paths_wrap_the_service_command(
    script: str,
    function_start: str,
    function_end: str,
    runner_variable: str,
    service: str,
    child_command: str,
):
    script_text = _read(script)
    function_text = _between(script_text, function_start, function_end)

    assert re.search(
        rf"(?m)^{runner_variable}=.*rotating_log_runner\.py",
        script_text,
    )
    assert "--init" in function_text
    assert re.search(
        rf"\$log_runner_host:\${re.escape(runner_variable)}:ro",
        function_text,
    )
    _assert_runner_precedes(function_text, child_command, runner_variable)
    assert re.search(rf"--service(?:[=\s'\"])+{re.escape(service)}\b", function_text)


def test_build_triton_generated_dockerfile_bundles_and_runs_through_runner():
    script = _read("scripts/bash/build_triton.sh")
    function_text = _between(
        script,
        "generate_dockerfile() {",
        "# ── Argument parsing ──",
    )

    _assert_generated_dockerfile_wraps_triton(function_text)


def test_lib_triton_temporary_build_context_bundles_and_runs_through_runner():
    script = _read("scripts/bash/lib/triton.sh")
    function_text = _between(
        script,
        "build_triton_image() {",
        "#  triton_resolve_container_name",
    )

    # This builder uses a temporary Docker context, so the runner must be staged
    # into it before the generated Dockerfile can COPY it into the image.
    staging_lines = [
        line
        for line in function_text.splitlines()
        if "rotating_log_runner.py" in line and "$build_dir" in line
    ]
    assert staging_lines, "runner is not staged into the temporary Docker context"
    _assert_generated_dockerfile_wraps_triton(function_text)
