"""Tests for runtime fingerprint probing."""

from __future__ import annotations

import subprocess

from engine.runtime import fingerprint


def test_probe_gpu_via_nvidia_smi_parses_compute_cap_and_name(monkeypatch):
    monkeypatch.setattr(fingerprint.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def fake_check_output(args, **kwargs):
        assert args[:2] == ["nvidia-smi", "--id=0"]
        assert "--query-gpu=compute_cap,name" in args
        return "8.9, NVIDIA GeForce RTX 4090\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    assert fingerprint._probe_gpu_via_nvidia_smi(0) == (
        "sm_89",
        "NVIDIA GeForce RTX 4090",
    )


def test_probe_current_environment_uses_nvidia_smi_before_torch(monkeypatch):
    monkeypatch.setattr(
        fingerprint,
        "_probe_gpu_via_nvidia_smi",
        lambda device_index: ("sm_89", "NVIDIA GeForce RTX 4090"),
    )

    def fail_torch_probe(device_index):
        raise AssertionError("torch probe should not be used when nvidia-smi succeeds")

    monkeypatch.setattr(fingerprint, "_probe_gpu_via_torch", fail_torch_probe)
    monkeypatch.setattr(fingerprint, "_probe_tensorrt_version", lambda: "10.9.0")
    monkeypatch.setattr(fingerprint, "_probe_driver_via_nvml", lambda: "570.153.02")

    env = fingerprint.probe_current_environment(0)

    assert env.gpu_sm == "sm_89"
    assert env.gpu_name == "NVIDIA GeForce RTX 4090"
    assert env.tensorrt_version == "10.9.0"
    assert env.driver_version == "570.153.02"
