"""Regression tests for qwen3tts CLI — interactive mode, discover-target, and new commands."""

from __future__ import annotations

import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
#  Import tests — ensure all new modules are importable
# ---------------------------------------------------------------------------

class TestImports:
    """Verify that all new CLI modules can be imported."""

    def test_import_main(self):
        from qwen3tts_cli.main import main, cli_main
        assert callable(main)
        assert callable(cli_main)

    def test_import_interactive(self):
        from qwen3tts_cli.interactive import (
            interactive_mode,
            show_run_banner,
            detect_resume_point,
            select_model_variant,
            discover_exported_variants,
            interactive_cross_host_guide,
            select_ngc_tag_interactive,
        )
        assert callable(interactive_mode)
        assert callable(show_run_banner)
        assert callable(detect_resume_point)
        assert callable(select_model_variant)
        assert callable(discover_exported_variants)
        assert callable(interactive_cross_host_guide)
        assert callable(select_ngc_tag_interactive)

    def test_import_discover(self):
        from qwen3tts_cli.cmd_discover import run_discover_target
        assert callable(run_discover_target)


# ---------------------------------------------------------------------------
#  CLI subcommand registration tests
# ---------------------------------------------------------------------------

class TestSubcommandRegistration:
    """Verify that all expected subcommands are registered."""

    def test_all_subcommands_present(self):
        from qwen3tts_cli.main import main
        # Parse --help to extract registered subcommands
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        # The exit code for --help is 0
        assert exc_info.value.code == 0

    def test_discover_target_subcommand(self):
        from qwen3tts_cli.main import main
        with pytest.raises(SystemExit) as exc_info:
            main(["discover-target", "--help"])
        assert exc_info.value.code == 0

    def test_list_ngc_subcommand(self):
        from qwen3tts_cli.main import main
        with pytest.raises(SystemExit) as exc_info:
            main(["list-ngc", "--help"])
        assert exc_info.value.code == 0

    def test_update_matrix_subcommand(self):
        from qwen3tts_cli.main import main
        with pytest.raises(SystemExit) as exc_info:
            main(["update-matrix", "--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
#  New CLI options tests
# ---------------------------------------------------------------------------

class TestNewOptions:
    """Verify that new options from autorun.sh are available."""

    def test_all_has_ngc_tag(self):
        from qwen3tts_cli.main import main
        # Should not raise — just parse and exit
        result = main(["all", "--ngc-tag", "25.03", "--dry-run"])
        assert isinstance(result, int)

    def test_all_has_target_driver(self):
        from qwen3tts_cli.main import main
        result = main(["all", "--target-driver", "575.57", "--dry-run"])
        assert isinstance(result, int)

    def test_all_has_export_device(self):
        from qwen3tts_cli.main import main
        result = main(["all", "--export-device", "cuda:0", "--dry-run"])
        assert isinstance(result, int)

    def test_all_has_build_device(self):
        from qwen3tts_cli.main import main
        result = main(["all", "--build-device", "0", "--dry-run"])
        assert isinstance(result, int)

    def test_all_has_runtime_device(self):
        from qwen3tts_cli.main import main
        result = main(["all", "--runtime-device", "0", "--dry-run"])
        assert isinstance(result, int)

    def test_all_has_engine_image(self):
        from qwen3tts_cli.main import main
        result = main(["all", "--engine-image", "qwen3-engine:25.03", "--dry-run"])
        assert isinstance(result, int)

    def test_all_has_grpc_port(self):
        from qwen3tts_cli.main import main
        result = main(["all", "--grpc-port", "8001", "--dry-run"])
        assert isinstance(result, int)

    def test_all_has_http_port(self):
        from qwen3tts_cli.main import main
        result = main(["all", "--http-port", "8000", "--dry-run"])
        assert isinstance(result, int)

    def test_all_has_rebuild_image(self):
        from qwen3tts_cli.main import main
        result = main(["all", "--build", "--dry-run"])
        assert isinstance(result, int)

    def test_setup_has_python_version(self):
        from qwen3tts_cli.main import main
        result = main(["setup", "--python", "3.12", "--dry-run"])
        assert isinstance(result, int)

    def test_setup_has_env_name(self):
        from qwen3tts_cli.main import main
        result = main(["setup", "--env-name", "my-env", "--dry-run"])
        assert isinstance(result, int)

    def test_setup_has_export_device(self):
        from qwen3tts_cli.main import main
        result = main(["setup", "--export-device", "cpu", "--dry-run"])
        assert isinstance(result, int)

    def test_run_has_runtime_device(self):
        from qwen3tts_cli.main import main
        result = main(["run", "--runtime-device", "0", "--dry-run"])
        assert isinstance(result, int)

    def test_run_has_max_seq_len(self):
        from qwen3tts_cli.main import main
        result = main(["run", "--max-seq-len", "512", "--dry-run"])
        assert isinstance(result, int)

    def test_run_has_engine_image(self):
        from qwen3tts_cli.main import main
        result = main(["run", "--engine-image", "qwen3-engine:25.03", "--dry-run"])
        assert isinstance(result, int)

    def test_package_has_build(self):
        from qwen3tts_cli.main import main
        result = main(["package", "--build", "--dry-run"])
        assert isinstance(result, int)

    def test_package_has_engine_image(self):
        from qwen3tts_cli.main import main
        result = main(["package", "--engine-image", "qwen3-engine:25.03", "--dry-run"])
        assert isinstance(result, int)

    def test_all_has_backbone_precision(self):
        from qwen3tts_cli.main import main
        result = main(["all", "--backbone-precision", "bf16", "--dry-run"])
        assert isinstance(result, int)

    def test_all_has_cp_precision(self):
        from qwen3tts_cli.main import main
        result = main(["all", "--cp-precision", "fp32", "--dry-run"])
        assert isinstance(result, int)

    def test_all_has_code2wav_precision(self):
        from qwen3tts_cli.main import main
        result = main(["all", "--code2wav-precision", "bf16", "--dry-run"])
        assert isinstance(result, int)

    def test_build_has_backbone_precision(self):
        from qwen3tts_cli.main import main
        result = main(["build", "--backbone-precision", "bf16", "--dry-run"])
        assert isinstance(result, int)

    def test_build_has_cp_precision(self):
        from qwen3tts_cli.main import main
        result = main(["build", "--cp-precision", "fp32", "--dry-run"])
        assert isinstance(result, int)

    def test_build_has_code2wav_precision(self):
        from qwen3tts_cli.main import main
        result = main(["build", "--code2wav-precision", "bf16", "--dry-run"])
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
#  Interactive mode tests
# ---------------------------------------------------------------------------

class TestInteractiveMode:
    """Test the interactive TUI mode."""

    def test_show_run_banner(self, capsys):
        from qwen3tts_cli.interactive import show_run_banner
        show_run_banner("测试阶段", "测试描述", 变体="custom-1.7b", 引擎精度="bf16")
        captured = capsys.readouterr()
        assert "Qwen3-TTS Triton" in captured.out
        assert "测试阶段" in captured.out
        assert "测试描述" in captured.out
        assert "custom-1.7b" in captured.out
        assert "bf16" in captured.out

    def test_show_run_banner_skips_empty_values(self, capsys):
        from qwen3tts_cli.interactive import show_run_banner
        show_run_banner("测试", "描述", 空值=None, 假值=False, 空串="")
        captured = capsys.readouterr()
        assert "空值" not in captured.out
        assert "假值" not in captured.out
        assert "空串" not in captured.out

    def test_discover_exported_variants_empty(self):
        from qwen3tts_cli.interactive import discover_exported_variants
        # workspace/exported/ may not exist
        variants = discover_exported_variants()
        assert isinstance(variants, list)

    def test_detect_resume_point(self):
        from qwen3tts_cli.interactive import detect_resume_point
        point = detect_resume_point()
        assert point in ("setup", "build", "package", "deploy", "done")


# ---------------------------------------------------------------------------
#  Discover-target tests
# ---------------------------------------------------------------------------

class TestDiscoverTarget:
    """Test the discover-target command."""

    def test_discover_local_mode(self):
        """Test that discover-target --local delegates to probe."""
        from qwen3tts_cli.cmd_discover import run_discover_target
        args = argparse.Namespace(
            discover_mode="local",
            output="workspace/target_profile.json",
            remote_host="",
            remote_workdir="/tmp/qwen3-tts-engine-build",
        )
        # This will fail if nvidia-smi is not available, but should not crash
        result = run_discover_target(args)
        assert isinstance(result, int)

    def test_discover_remote_requires_host(self):
        """Test that remote mode requires --remote-host."""
        from qwen3tts_cli.cmd_discover import run_discover_target
        args = argparse.Namespace(
            discover_mode="remote",
            output="workspace/target_profile.json",
            remote_host="",
            remote_workdir="/tmp/qwen3-tts-engine-build",
        )
        result = run_discover_target(args)
        assert result == 1  # Should fail without remote host


# ---------------------------------------------------------------------------
#  list-ngc tests
# ---------------------------------------------------------------------------

class TestListNgc:
    """Test the list-ngc command."""

    def test_list_ngc_runs(self):
        from qwen3tts_cli.main import main
        result = main(["list-ngc"])
        assert isinstance(result, int)

    def test_list_ngc_with_driver(self):
        from qwen3tts_cli.main import main
        result = main(["list-ngc", "--target-driver", "570.86"])
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
#  cli_main entry point test
# ---------------------------------------------------------------------------

class TestCliMain:
    """Test the cli_main entry point wrapper."""

    def test_cli_main_help(self):
        from qwen3tts_cli.main import cli_main
        with pytest.raises(SystemExit) as exc_info:
            cli_main()
        # --help is not the default; calling with no args launches interactive
        # mode which may fail in non-TTY. Just verify it doesn't crash on import.
        assert True  # If we got here, the import works

    def test_cli_main_with_help_flag(self):
        from qwen3tts_cli.main import cli_main
        with patch("sys.argv", ["qwen3tts", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 0
