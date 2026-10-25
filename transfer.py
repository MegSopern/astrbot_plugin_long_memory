"""导入导出：打包成 zip / 从 zip 读回。

安全原则：**全程不落盘解压**。zip 只读进内存，逐条目校验后交给 store 按规范 JSON 写盘，
所以 zip-slip、符号链接、设备文件这些在结构上就无法生效，下面的检查是纵深防御。
"""

from __future__ import annotations

import json
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any

try:  # AstrBot 用包路径导入；单测走顶层导入
    from .store import ARCHIVE_FORMAT, SCHEMA_VERSION, UID_RE, normalize_items
except ImportError:  # pragma: no cover
    from store import (  # type: ignore[no-redef]
        ARCHIVE_FORMAT,
        SCHEMA_VERSION,
        UID_RE,
        normalize_items,
    )

FORMAT = ARCHIVE_FORMAT
FORMAT_VERSION = 1
MANIFEST = "manifest.json"
MAX_ENTRIES = 2000
MAX_ENTRY_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_RATIO = 200


class ArchiveError(Exception):
    """包不合法，整包拒绝。"""


# ── 条目校验 ─────────────────────────────────────────────────────────
def _norm(name: str) -> str:
    return (name or "").replace("\\", "/")


def _safe_name(info: zipfile.ZipInfo) -> str | None:
    """返回规范条目名；目录条目返回 None；不合法抛 ArchiveError。"""
    raw = _norm(info.filename)
    if not raw:
        raise ArchiveError("空的条目名")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise ArchiveError(f"不允许绝对路径：{raw}")

    segs = raw.split("/")
    if segs and segs[-1] == "":
        segs = segs[:-1]
    if not segs:
        raise ArchiveError("空的条目名")
    if any(seg in ("", ".", "..") for seg in segs):
        raise ArchiveError(f"不允许路径穿越：{raw}")

    if len(segs) == 1 and info.is_dir() and segs[0] == "memory":
        return None
    if len(segs) == 1 and segs[0] == MANIFEST and not info.is_dir():
        return MANIFEST
    if len(segs) == 2 and segs[0] == "memory" and segs[1].endswith(".json"):
        if UID_RE.match(segs[1][:-5]):
            return f"memory/{segs[1]}"
    raise ArchiveError(f"不认识的条目：{raw}")


def _check_entry(info: zipfile.ZipInfo) -> None:
    if info.flag_bits & 0x1:
        raise ArchiveError(f"不支持加密条目：{info.filename}")
    if (info.external_attr >> 16) & 0o170000 == 0o120000:
        raise ArchiveError(f"不允许符号链接：{info.filename}")
    if info.file_size > MAX_ENTRY_BYTES:
        raise ArchiveError(f"单条过大：{info.filename}")
    if info.file_size > 1024 * 1024 and info.compress_size:
        if info.file_size / max(1, info.compress_size) > MAX_RATIO:
            raise ArchiveError(f"压缩比异常，疑似 zip 炸弹：{info.filename}")


def _read_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    data = zf.read(info)
    if len(data) != info.file_size:
        raise ArchiveError(f"条目长度与声明不符：{info.filename}")
    return data


# ── 导出 ─────────────────────────────────────────────────────────────
def build_archive(
    dest: str | Path,
    data: dict[str, list[dict[str, Any]]],
    *,
    exported_by: str,
    tmp_dir: str | Path | None = None,
) -> dict[str, Any]:
    """打一个可导入的包，返回 manifest。"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tmp_dir) if tmp_dir else dest.parent / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp = tmp_root / f"{dest.name}.{os.getpid()}.tmp"

    users = []
    total_items = 0
    for uid in sorted(data):
        items = normalize_items(data[uid])
        if not items:
            continue
        total_items += len(items)
        users.append({
            "uid": uid,
            "items": len(items),
            "updated_at": max(int(m.get("updated_at", 0)) for m in items),
        })

    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "plugin": "astrbot_plugin_long_memory",
        "exported_at": int(time.time()),
        "exported_by": str(exported_by),
        "partial": False,
        "counts": {"users": len(users), "items": total_items},
        "users": users,
    }

    stamp = time.localtime()[:6]
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr(zipfile.ZipInfo(MANIFEST, stamp), json.dumps(manifest, ensure_ascii=False, indent=2))
        for entry in users:
            uid = entry["uid"]
            payload = {
                "v": SCHEMA_VERSION,
                "uid": uid,
                "items": normalize_items(data[uid]),
                "updated_at": entry["updated_at"],
            }
            zf.writestr(
                zipfile.ZipInfo(f"memory/{uid}.json", stamp),
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
    os.replace(tmp, dest)
    return manifest


# ── 导入 ─────────────────────────────────────────────────────────────
def read_archive(
    path: str | Path, *, max_mb: int = 32
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, int]]:
    """校验并读出包内容。返回 (manifest, {uid: items}, stats)。"""
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise ArchiveError(f"读不到文件：{exc}") from exc
    if size > max(1, int(max_mb)) * 1024 * 1024:
        raise ArchiveError(f"包太大：{size / 1048576:.1f}MB")
    if not zipfile.is_zipfile(source):
        raise ArchiveError("不是有效的 zip 文件")

    try:
        zf = zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise ArchiveError("zip 已损坏") from exc

    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_ENTRIES:
            raise ArchiveError(f"条目过多：{len(infos)}")

        seen: set[str] = set()
        targets: dict[str, zipfile.ZipInfo] = {}
        total = 0
        for info in infos:
            name = _safe_name(info)
            if name is None:
                continue
            _check_entry(info)
            if name in seen:
                raise ArchiveError(f"重复条目：{name}")
            seen.add(name)
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise ArchiveError("解压后体积过大")
            targets[name] = info

        if MANIFEST not in targets:
            raise ArchiveError("缺少 manifest.json，不是本插件导出的包")
        try:
            manifest = json.loads(_read_entry(zf, targets[MANIFEST]).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ArchiveError("manifest.json 不是合法 JSON") from exc
        if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
            raise ArchiveError("不是本插件导出的包")
        version = manifest.get("format_version")
        if isinstance(version, bool) or not isinstance(version, int) or version > FORMAT_VERSION:
            raise ArchiveError("包格式版本过新，请升级插件")

        data: dict[str, list[dict[str, Any]]] = {}
        dropped = 0
        for name, info in targets.items():
            if not name.startswith("memory/"):
                continue
            uid = name[len("memory/"):-len(".json")]
            try:
                obj = json.loads(_read_entry(zf, info).decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ArchiveError(f"memory/{uid}.json 不是合法 JSON") from exc
            if not isinstance(obj, dict) or not isinstance(obj.get("items"), list):
                raise ArchiveError(f"memory/{uid}.json 结构不对")
            good = normalize_items(obj["items"])
            dropped += len(obj["items"]) - len(good)
            data[uid] = good

    return manifest, data, {"dropped_items": dropped}
