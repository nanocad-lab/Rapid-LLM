#!/usr/bin/env python3
# Copyright 2026 NanoCad lab, UCLA
# https://nanocad.ee.ucla.edu/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Modify or insert a scalar field in YAML configs (PyYAML only, minimal text edit).

- If the dotted path exists, overwrite its scalar value (preserving inline comment).
- If it does not exist, create it (with optional comment) under existing ancestor mappings.
- Skips files where an ancestor is not a mapping.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modify/insert a YAML field.")
    parser.add_argument("--scope", choices=["hardware", "software"], required=True, help="which config family to update")
    parser.add_argument("--path", required=True, help="dotted path to set (e.g., network.overlap.tp_overlap)")
    parser.add_argument("--value", required=True, help='YAML literal for the value (e.g., 0.0, true, "text")')
    parser.add_argument("--comment", default="", help="comment to place above the inserted field (add-only)")
    parser.add_argument("--dry-run", action="store_true", help="print planned changes without writing files")
    return parser.parse_args()


def parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def find_roots(scope: str) -> List[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    if scope == "hardware":
        roots = [
            repo_root / "configs" / "hardware-config",
            repo_root / "validation_scripts" / "validation_configs" / "hardware-config",
        ]
    else:
        roots = [
            repo_root / "configs" / "model-config",
            repo_root / "validation_scripts" / "validation_configs" / "model-config",
        ]
    return [p for p in roots if p.exists()]


def key_from_line(line: str) -> Optional[Tuple[int, str]]:
    if ":" not in line:
        return None
    stripped = line.split("#", 1)[0].rstrip()
    if ":" not in stripped:
        return None
    prefix, _ = stripped.split(":", 1)
    if not prefix.strip():
        return None
    indent = len(prefix) - len(prefix.lstrip(" "))
    key = prefix.strip()
    return indent, key


def locate_block(lines: List[str], path_parts: List[str]) -> Optional[Tuple[int, int]]:
    idx = 0
    indent = 0
    last_idx = None
    last_indent = None
    for part in path_parts:
        found = False
        for i in range(idx, len(lines)):
            res = key_from_line(lines[i])
            if res is None:
                continue
            ind, key = res
            if ind < indent:
                return None
            if ind == indent and key == part:
                last_idx, last_indent = i, ind
                idx = i + 1
                indent = ind + 2
                found = True
                break
        if not found:
            return None
    if last_idx is None or last_indent is None:
        return None
    return last_idx, last_indent


def find_insertion_index(lines: List[str], parent_line_idx: int, parent_indent: int) -> int:
    for i in range(parent_line_idx + 1, len(lines)):
        res = key_from_line(lines[i])
        if res is None:
            continue
        ind, _ = res
        if ind <= parent_indent:
            return i
    return len(lines)


def render_scalar(value: Any) -> str:
    dumped = yaml.safe_dump(value, default_flow_style=True)
    # Take only the first line to avoid document end marker ('...')
    first_line = dumped.split('\n', 1)[0]
    return first_line


def ancestors_are_mappings(data: Dict[str, Any], path_parts: List[str]) -> bool:
    cursor = data
    for part in path_parts[:-1]:
        if not isinstance(cursor, dict):
            return False
        if part not in cursor:
            return True
        cursor = cursor[part]
    return isinstance(cursor, dict)


def insert_block(lines: List[str], path_parts: List[str], value: Any, comment: str) -> List[str]:
    # Find deepest existing ancestor
    deepest_idx = -1
    deepest_indent = -2
    missing_at = 0
    for i in range(len(path_parts)):
        loc = locate_block(lines, path_parts[: i + 1])
        if loc is None:
            missing_at = i
            break
        deepest_idx, deepest_indent = loc
        missing_at = i + 1

    # All parts exist
    if missing_at >= len(path_parts):
        return lines

    missing_parts = path_parts[missing_at:]
    insertion_idx = find_insertion_index(lines, deepest_idx, deepest_indent) if deepest_idx >= 0 else len(lines)
    indent = deepest_indent + 2

    snippet_lines: List[str] = []
    for depth, key in enumerate(missing_parts):
        pad = " " * (indent + depth * 2)
        is_leaf = depth == len(missing_parts) - 1
        if is_leaf:
            if comment:
                snippet_lines.append(f"{pad}# {comment}")
            snippet_lines.append(f"{pad}{key}: {render_scalar(value)}")
        else:
            snippet_lines.append(f"{pad}{key}:")

    return lines[:insertion_idx] + snippet_lines + lines[insertion_idx:]


def update_leaf_value(lines: List[str], path_parts: List[str], value: Any) -> List[str]:
    loc = locate_block(lines, path_parts)
    if loc is None:
        return lines
    leaf_idx, leaf_indent = loc
    key = path_parts[-1]
    line = lines[leaf_idx]
    comment = ""
    if "#" in line:
        _, comment = line.split("#", 1)
    new_line = f"{' ' * leaf_indent}{key}: {render_scalar(value)}"
    if comment:
        new_line += f"  #{comment.lstrip()}"
    lines[leaf_idx] = new_line
    return lines


def set_field(path: Path, dotted_path: str, value: Any, comment: str = "", dry_run: bool = False) -> str:
    path_parts = dotted_path.split(".")
    try:
        text = path.read_text()
    except Exception as exc:
        return f"[SKIP] {path}: read error: {exc}"

    lines = [ln for ln in text.splitlines() if ln.strip() != "..."]

    try:
        data = yaml.safe_load("\n".join(lines)) or {}
    except Exception as exc:
        return f"[SKIP] {path}: yaml parse error: {exc}"

    if not isinstance(data, dict):
        return f"[SKIP] {path}: top-level is not a mapping"

    if not ancestors_are_mappings(data, path_parts):
        return f"[SKIP] {path}: ancestor path is missing or not a mapping; not inserting"

    exists = True
    cursor: Any = data
    for part in path_parts:
        if isinstance(cursor, dict) and part in cursor:
            cursor = cursor[part]
        else:
            exists = False
            break

    if exists:
        new_lines = update_leaf_value(lines, path_parts, value)
    else:
        new_lines = insert_block(lines, path_parts, value, comment)

    new_text = "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")
    if dry_run:
        return f"[DRY-RUN] {path}"
    try:
        path.write_text(new_text)
    except Exception as exc:
        return f"[SKIP] {path}: write error: {exc}"
    return f"[UPDATED] {path}"


def main() -> None:
    args = parse_args()
    value = parse_value(args.value)
    roots = find_roots(args.scope)
    path_parts = args.path
    modified = 0
    skipped = 0
    for root in roots:
        for yaml_path in sorted(root.rglob("*.yaml")):
            result = set_field(yaml_path, path_parts, value, args.comment, args.dry_run)
            if result.startswith("[UPDATED]") or result.startswith("[DRY-RUN]"):
                modified += 1
            else:
                skipped += 1
            print(result)
    print(f"Done. Modified: {modified}, skipped: {skipped}")


if __name__ == "__main__":
    main()
