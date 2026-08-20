"""Contract tests for the GitLab Release publisher.

The real API is intentionally replaced with a small ``glab`` test double. This
keeps the release path locally testable while still exercising its shell
validation, command ordering, form encoding, and response verification.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts/bash/create_gitlab_release.sh"


def _release_environment(tmp_path: Path) -> dict[str, str]:
    tag = "v0.2.0a13"
    project_url = "http://gitlab.example.test/group/project"
    wheel = "qwen3_tts_client-0.2.0a13-py3-none-any.whl"
    browser = "xmultimodalinteraction-qwen3tts-browser-0.2.0-alpha.13.tgz"
    demo = "qwen3tts-demo-0.2.0-alpha.13.tar.gz"
    return {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "GLAB_CALL_LOG": str(tmp_path / "glab-calls.log"),
        "CI_COMMIT_SHA": "a" * 40,
        "CI_COMMIT_TAG": tag,
        "CI_PROJECT_ID": "844",
        "CI_PROJECT_PATH": "group/project",
        "CI_PROJECT_URL": project_url,
        "CI_SERVER_FQDN": "gitlab.example.test",
        "CI_SERVER_PROTOCOL": "http",
        "ENGINE_RELEASE_IMAGE": f"registry.example.test/engine:{tag}",
        "TRITON_RELEASE_IMAGE": f"registry.example.test/triton:{tag}",
        "WHEEL_FILENAME": wheel,
        "WHEEL_REGISTRY_URL": f"{project_url}/packages/{wheel}",
        "WHEEL_RELEASE_URL": (
            f"{project_url}/-/releases/{tag}/downloads/client-sdk/{wheel}"
        ),
        "WHEEL_SHA256": "b" * 64,
        "BROWSER_SDK_TARBALL": browser,
        "BROWSER_SDK_GENERIC_URL": f"{project_url}/packages/{browser}",
        "DEMO_ARCHIVE": demo,
        "DEMO_ARCHIVE_URL": f"{project_url}/packages/{demo}",
    }


def _write_fake_glab(tmp_path: Path) -> Path:
    fake = tmp_path / "glab"
    fake.write_text(
        """#!/bin/sh
set -eu

{
  echo CALL
  printf 'ARG=%s\n' "$@"
} >> "$GLAB_CALL_LOG"

case "${1:-}" in
  config)
    exit 0
    ;;
  release)
    [ "${2:-}" = create ]
    exit 0
    ;;
  api)
    shift
    method=GET
    name=
    url=
    direct_path=
    endpoint=
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --hostname)
          shift 2
          ;;
        --method)
          method=$2
          shift 2
          ;;
        --form)
          case "$2" in
            name=*) name=${2#name=} ;;
            url=*) url=${2#url=} ;;
            direct_asset_path=*) direct_path=${2#direct_asset_path=} ;;
          esac
          shift 2
          ;;
        --silent)
          shift
          ;;
        *)
          endpoint=$1
          shift
          ;;
      esac
    done
    [ "$endpoint" != job ] || exit 0
    if [ "$method" = POST ]; then
      [ -n "$name" ] && [ -n "$url" ]
      case "$direct_path" in /*) ;; *) exit 3 ;; esac
      direct_url="${CI_PROJECT_URL}/-/releases/${CI_COMMIT_TAG}/downloads${direct_path}"
      jq -cn \
        --arg name "$name" \
        --arg url "$url" \
        --arg direct_url "$direct_url" \
        '{name: $name, url: $url, direct_asset_url: $direct_url, link_type: "package"}'
    else
      printf '[]\n'
    fi
    ;;
  *)
    echo "Unexpected glab command: $*" >&2
    exit 2
    ;;
esac
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake


def test_missing_release_metadata_fails_before_any_gitlab_request(tmp_path: Path):
    _write_fake_glab(tmp_path)
    environment = _release_environment(tmp_path)
    environment.pop("WHEEL_FILENAME")

    result = subprocess.run(
        ["sh", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "WHEEL_FILENAME" in result.stderr
    assert not Path(environment["GLAB_CALL_LOG"]).exists()


def test_release_links_use_nonempty_multipart_form_contract(tmp_path: Path):
    _write_fake_glab(tmp_path)
    environment = _release_environment(tmp_path)

    subprocess.run(
        ["sh", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    calls = Path(environment["GLAB_CALL_LOG"]).read_text(encoding="utf-8")
    assert "ARG=--raw-field" not in calls
    assert calls.count("ARG=--form") == 12
    assert calls.count("ARG=--method\nARG=POST") == 3
    assert f"ARG=name={environment['WHEEL_FILENAME']}" in calls
    assert f"ARG=direct_asset_path=/client-sdk/{environment['WHEEL_FILENAME']}" in calls
    assert f"ARG=name={environment['BROWSER_SDK_TARBALL']}" in calls
    assert f"ARG=name={environment['DEMO_ARCHIVE']}" in calls
    assert "ARG=release\nARG=create" in calls


def test_release_job_downloads_each_originating_dotenv_artifact():
    import yaml

    config = yaml.safe_load((REPO_ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8"))
    needs = {
        entry["job"]: entry.get("artifacts")
        for entry in config["create-release"]["needs"]
    }

    assert needs["publish-client-wheel"] is True
    assert needs["build-web-release"] is True
    assert needs["publish-browser-sdk"] is True
    assert needs["build-engine-image"] is False
