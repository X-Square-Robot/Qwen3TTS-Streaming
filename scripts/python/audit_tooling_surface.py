#!/usr/bin/env python3
"""Summarize the repository's scripts/tests surface for governance work."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

PYTHON_ROOTS = ("scripts", "tests")
COMMON_DUPLICATE_NAMES = {"__init__", "callback", "forward", "main", "parse_args"}
MANAGED_SYS_PATH_BOOTSTRAPS = {
    "scripts/python/common.py",
    "tools/validation/_bootstrap.py",
}


@dataclass
class DuplicateSymbol:
    name: str
    count: int
    locations: list[str]


def _iter_files(base: Path, suffixes: Iterable[str]) -> list[Path]:
    suffix_set = set(suffixes)
    return sorted(
        path
        for path in base.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in suffix_set
    )


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _is_entrypoint(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        try:
            expr = ast.unparse(node.test)
        except Exception:
            continue
        if "__name__" in expr and "__main__" in expr:
            return True
    return False


def _top_duplicates(
    symbols: dict[str, list[str]],
    *,
    limit: int,
    min_count: int,
    include_common: bool,
) -> list[DuplicateSymbol]:
    rows: list[DuplicateSymbol] = []
    for name, locations in sorted(symbols.items()):
        if len(locations) < min_count:
            continue
        if not include_common and name in COMMON_DUPLICATE_NAMES:
            continue
        rows.append(
            DuplicateSymbol(
                name=name,
                count=len(locations),
                locations=locations[:12],
            )
        )
    rows.sort(key=lambda row: (-row.count, row.name))
    return rows[:limit]


def build_report(
    *,
    top_duplicates: int,
    min_duplicate_count: int,
    include_common_names: bool,
) -> dict[str, object]:
    python_files = _iter_files(REPO_ROOT / "scripts", [".py"]) + _iter_files(REPO_ROOT / "tests", [".py"])
    shell_files = _iter_files(REPO_ROOT / "scripts", [".sh"]) + _iter_files(REPO_ROOT / "tests", [".sh"])
    markdown_files = _iter_files(REPO_ROOT / "scripts", [".md"]) + _iter_files(REPO_ROOT / "tests", [".md"])
    engine_python_files = _iter_files(REPO_ROOT / "engine", [".py"])
    pycache_dirs = sorted(
        _relative(path)
        for root_name in PYTHON_ROOTS
        for path in (REPO_ROOT / root_name).rglob("__pycache__")
        if path.is_dir()
    )

    entrypoints: list[str] = []
    sys_path_bootstraps: list[str] = []
    managed_sys_path_bootstraps: list[str] = []
    unmanaged_sys_path_bootstraps: list[str] = []
    tools_importing_pytest_modules: list[str] = []
    non_e2e_imports_of_e2e_tests: list[str] = []
    function_defs: dict[str, list[str]] = defaultdict(list)
    class_defs: dict[str, list[str]] = defaultdict(list)
    per_dir_counter: Counter[str] = Counter()

    for path in python_files:
        rel = _relative(path)
        per_dir_counter[str(path.parent.relative_to(REPO_ROOT))] += 1

        source = path.read_text(encoding="utf-8")
        if "sys.path.insert" in source:
            sys_path_bootstraps.append(rel)
            if rel in MANAGED_SYS_PATH_BOOTSTRAPS:
                managed_sys_path_bootstraps.append(rel)
            else:
                unmanaged_sys_path_bootstraps.append(rel)
        if rel.startswith("tools/validation/") and "from tests.e2e.test_" in source:
            tools_importing_pytest_modules.append(rel)
        if rel.startswith("tests/") and not rel.startswith("tests/e2e/") and "from tests.e2e.test_" in source:
            non_e2e_imports_of_e2e_tests.append(rel)

        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            function_defs["<parse-error>"].append(f"{rel}:{exc.lineno}")
            continue

        if _is_entrypoint(tree):
            entrypoints.append(rel)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_defs[node.name].append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.ClassDef):
                class_defs[node.name].append(f"{rel}:{node.lineno}")

    report = {
        "repo_root": str(REPO_ROOT),
        "counts": {
            "engine_python_files": len(engine_python_files),
            "scripts_and_tests_python_files": len(python_files),
            "scripts_and_tests_shell_files": len(shell_files),
            "scripts_and_tests_markdown_files": len(markdown_files),
            "python_entrypoints": len(entrypoints),
            "sys_path_bootstraps": len(sys_path_bootstraps),
            "managed_sys_path_bootstraps": len(managed_sys_path_bootstraps),
            "unmanaged_sys_path_bootstraps": len(unmanaged_sys_path_bootstraps),
            "pycache_directories": len(pycache_dirs),
        },
        "per_directory_python_files": dict(sorted(per_dir_counter.items())),
        "entrypoints": entrypoints,
        "sys_path_bootstraps": sys_path_bootstraps,
        "managed_sys_path_bootstraps": managed_sys_path_bootstraps,
        "unmanaged_sys_path_bootstraps": unmanaged_sys_path_bootstraps,
        "tools_importing_pytest_modules": sorted(tools_importing_pytest_modules),
        "non_e2e_imports_of_e2e_tests": sorted(non_e2e_imports_of_e2e_tests),
        "pycache_directories": pycache_dirs,
        "top_duplicate_functions": [
            asdict(row)
            for row in _top_duplicates(
                function_defs,
                limit=top_duplicates,
                min_count=min_duplicate_count,
                include_common=include_common_names,
            )
        ],
        "top_duplicate_classes": [
            asdict(row)
            for row in _top_duplicates(
                class_defs,
                limit=top_duplicates,
                min_count=min_duplicate_count,
                include_common=include_common_names,
            )
        ],
    }
    return report


def _render_text(report: dict[str, object]) -> str:
    counts = report["counts"]
    lines = [
        "Tooling surface report",
        f"repo_root: {report['repo_root']}",
        "",
        "Counts:",
        f"  engine_python_files: {counts['engine_python_files']}",
        f"  scripts_and_tests_python_files: {counts['scripts_and_tests_python_files']}",
        f"  scripts_and_tests_shell_files: {counts['scripts_and_tests_shell_files']}",
        f"  scripts_and_tests_markdown_files: {counts['scripts_and_tests_markdown_files']}",
        f"  python_entrypoints: {counts['python_entrypoints']}",
        f"  sys_path_bootstraps: {counts['sys_path_bootstraps']}",
        f"  managed_sys_path_bootstraps: {counts['managed_sys_path_bootstraps']}",
        f"  unmanaged_sys_path_bootstraps: {counts['unmanaged_sys_path_bootstraps']}",
        f"  pycache_directories: {counts['pycache_directories']}",
        "",
        "Python files by directory:",
    ]
    for name, count in report["per_directory_python_files"].items():
        lines.append(f"  {name}: {count}")

    lines.append("")
    lines.append("Top duplicate functions:")
    for row in report["top_duplicate_functions"]:
        lines.append(f"  {row['name']}: {row['count']}")
        for location in row["locations"]:
            lines.append(f"    - {location}")

    lines.append("")
    lines.append("Top duplicate classes:")
    for row in report["top_duplicate_classes"]:
        lines.append(f"  {row['name']}: {row['count']}")
        for location in row["locations"]:
            lines.append(f"    - {location}")

    lines.append("")
    lines.append("unmanaged sys.path bootstraps:")
    for path in report["unmanaged_sys_path_bootstraps"]:
        lines.append(f"  - {path}")

    if report["managed_sys_path_bootstraps"]:
        lines.append("")
        lines.append("managed sys.path bootstraps:")
        for path in report["managed_sys_path_bootstraps"]:
            lines.append(f"  - {path}")

    if report["tools_importing_pytest_modules"]:
        lines.append("")
        lines.append("tools/validation importing pytest modules:")
        for path in report["tools_importing_pytest_modules"]:
            lines.append(f"  - {path}")

    if report["non_e2e_imports_of_e2e_tests"]:
        lines.append("")
        lines.append("non-e2e files importing tests/e2e/test_*.py:")
        for path in report["non_e2e_imports_of_e2e_tests"]:
            lines.append(f"  - {path}")

    if report["pycache_directories"]:
        lines.append("")
        lines.append("__pycache__ directories under scripts/tests:")
        for path in report["pycache_directories"]:
            lines.append(f"  - {path}")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--top-duplicates",
        type=int,
        default=15,
        help="How many duplicate symbols to show per section.",
    )
    parser.add_argument(
        "--min-duplicate-count",
        type=int,
        default=2,
        help="Minimum number of occurrences required to report a duplicate symbol.",
    )
    parser.add_argument(
        "--include-common-names",
        action="store_true",
        help="Include noisy names such as main/parse_args/forward in duplicate reports.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        top_duplicates=args.top_duplicates,
        min_duplicate_count=args.min_duplicate_count,
        include_common_names=args.include_common_names,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
