"""存储层：文件版。一人一文件，落在 StarTools.get_data_dir() 指向的插件数据目录。

原子性：所有写入走 `.tmp` → fsync → os.replace，保证「要么旧的、要么新的」，
不会出现半截 JSON。Windows 上目标被占用时 os.replace 会 PermissionError，做退避重试。
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

UID_RE = re.compile(r"^\d{5,20}$")
SCHEMA_VERSION = 1
ARCHIVE_FORMAT = "astrbot-long-memory"  # 与 transfer.FORMAT 同源
KINDS = frozenset({"like", "dislike", "name", "city", "job", "habit"})
MAX_ITEM_TEXT = 40
MAX_NOBJ = 24
ACCEPT_ARCHIVE_MAX_BYTES = 64 * 1024 * 1024


# ── 基础工具 ─────────────────────────────────────────────────────────
def _now() -> int:
    return int(time.time())


def _atomic_write(path: Path, text: str, tmp_dir: Path) -> None:
    """写临时文件 → fsync → 原子替换。"""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    last: Exception | None = None
    for attempt in range(4):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:  # Windows: 目标被别的句柄占着
            last = exc
            time.sleep(0.05 * (attempt + 1))
    with contextlib.suppress(OSError):
        tmp.unlink()
    if last is not None:
        raise last


def _clamp_ts(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(0, min(_now() + 86400, value))


def normalize_item(raw: Any) -> dict[str, Any] | None:
    """把任意来源的一条记忆洗成合法结构；不合法返回 None。

    注意 hits：JSON 的 true 在 Python 里是 bool，而 bool 是 int 的子类，
    所以 `isinstance(True, int)` 为真 —— 必须显式排除，否则排序会静默错乱。
    """
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    if kind not in KINDS:
        return None
    text = raw.get("text")
    if not isinstance(text, str):
        return None
    text = text.strip()[:MAX_ITEM_TEXT]
    if not text:
        return None
    nobj = raw.get("nobj")
    nobj = (
        nobj.strip()[:MAX_NOBJ]
        if isinstance(nobj, str) and nobj.strip()
        else text.lower()
    )
    hits = raw.get("hits", 1)
    hits = (
        1
        if isinstance(hits, bool) or not isinstance(hits, int)
        else max(1, min(9999, hits))
    )
    created = _clamp_ts(raw.get("created_at"), _now())
    updated = _clamp_ts(raw.get("updated_at"), created)
    iid = raw.get("id")
    iid = (
        iid.strip()[:32]
        if isinstance(iid, str) and iid.strip()
        else uuid.uuid4().hex[:12]
    )
    return {
        "id": iid,
        "kind": kind,
        "text": text,
        "nobj": nobj,
        "hits": hits,
        "created_at": created,
        "updated_at": updated,
    }


def normalize_items(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        got = normalize_item(item)
        if got:
            out.append(got)
    return out


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── 记忆存储 ─────────────────────────────────────────────────────────
class MemoryStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.memory_dir = self.root / "memory"
        self.tmp_dir = self.root / ".tmp"
        self.exports_dir = self.root / "exports"
        self.backups_dir = self.root / "backups"
        self.bulk = asyncio.Lock()
        self._locks: dict[str, asyncio.Lock] = {}

    # ── 生命周期 ────────────────────────────────────────────────────
    def prepare(self) -> None:
        for d in (self.memory_dir, self.tmp_dir, self.exports_dir, self.backups_dir):
            d.mkdir(parents=True, exist_ok=True)
        cutoff = _now() - 3600
        for stale in self.tmp_dir.glob("*.tmp"):
            with contextlib.suppress(OSError):
                if stale.stat().st_mtime < cutoff:
                    stale.unlink()

    def lock(self, uid: str) -> asyncio.Lock:
        lock = self._locks.get(uid)
        if lock is None:
            if len(self._locks) > 2000:  # 只回收没在用的，避免破坏互斥
                idle = [k for k, v in self._locks.items() if not v.locked()][:1000]
                for k in idle:
                    self._locks.pop(k, None)
            lock = asyncio.Lock()
            self._locks[uid] = lock
        return lock

    # ── 单用户 ──────────────────────────────────────────────────────
    def path_of(self, uid: str) -> Path:
        if not UID_RE.match(uid or ""):
            raise ValueError(f"invalid uid: {uid!r}")
        return self.memory_dir / f"{uid}.json"

    def exists(self, uid: str) -> bool:
        try:
            return self.path_of(uid).is_file()
        except ValueError:
            return False

    def load_sync(self, uid: str) -> list[dict[str, Any]]:
        try:
            path = self.path_of(uid)
        except ValueError:
            return []
        try:
            raw = path.read_text("utf-8")
        except FileNotFoundError:
            return []
        except (OSError, UnicodeDecodeError):
            self._quarantine(path, "io")
            return []
        try:
            obj = json.loads(raw)
        except ValueError:
            self._quarantine(path, "json")
            return []
        if not isinstance(obj, dict) or not isinstance(obj.get("items"), list):
            self._quarantine(path, "shape")
            return []
        if obj.get("v") != SCHEMA_VERSION:
            # 未来版本：只忽略，不隔离 —— 不然降级跑一次就把新版数据改名了
            return []
        return normalize_items(obj["items"])

    async def load(self, uid: str) -> list[dict[str, Any]]:
        return self.load_sync(uid)

    def save_sync(self, uid: str, items: list[dict[str, Any]]) -> None:
        path = self.path_of(uid)
        payload = {
            "v": SCHEMA_VERSION,
            "uid": uid,
            "items": normalize_items(items),
            "updated_at": _now(),
        }
        _atomic_write(
            path, json.dumps(payload, ensure_ascii=False, indent=2), self.tmp_dir
        )

    async def save(self, uid: str, items: list[dict[str, Any]]) -> None:
        self.save_sync(uid, items)

    async def clear(self, uid: str) -> int:
        async with self.lock(uid):
            count = len(self.load_sync(uid))
            with contextlib.suppress(FileNotFoundError, ValueError):
                self.path_of(uid).unlink()
            return count

    def _quarantine(self, path: Path, why: str) -> None:
        dst = path.with_name(f"{path.name}.corrupt.{_now()}")
        with contextlib.suppress(OSError):
            path.replace(dst)

    # ── 批量 ────────────────────────────────────────────────────────
    def list_uids(self) -> list[str]:
        if not self.memory_dir.is_dir():
            return []
        out = []
        for path in self.memory_dir.glob("*.json"):
            if path.is_file() and UID_RE.match(path.stem):
                out.append(path.stem)
        return sorted(out)

    def load_all(self) -> dict[str, list[dict[str, Any]]]:
        data: dict[str, list[dict[str, Any]]] = {}
        for uid in self.list_uids():
            items = self.load_sync(uid)
            if items:
                data[uid] = items
        return data

    async def write_many(
        self, data: dict[str, list[dict[str, Any]]], *, delete_absent: bool
    ) -> tuple[int, int]:
        """写入包内数据；delete_absent 时删掉包内没有的用户。返回 (写入人数, 删除人数)。"""
        written = deleted = 0
        for uid, items in data.items():
            if not UID_RE.match(uid or ""):
                continue
            if items:
                self.save_sync(uid, items)
                written += 1
            elif delete_absent:
                with contextlib.suppress(FileNotFoundError):
                    self.path_of(uid).unlink()
                deleted += 1
        if delete_absent:
            for uid in self.list_uids():
                if uid not in data:
                    with contextlib.suppress(FileNotFoundError):
                        self.path_of(uid).unlink()
                    deleted += 1
        return written, deleted

    async def clear_all(self) -> dict[str, int]:
        stats = {"users": 0, "exports": 0, "backups": 0}
        async with self.bulk:
            for uid in self.list_uids():
                with contextlib.suppress(OSError):
                    self.path_of(uid).unlink()
                stats["users"] += 1
            for folder, key in (
                (self.exports_dir, "exports"),
                (self.backups_dir, "backups"),
            ):
                if folder.is_dir():
                    for path in folder.glob("*.zip"):
                        with contextlib.suppress(OSError):
                            path.unlink()
                        stats[key] += 1
            for stale in self.tmp_dir.glob("*.tmp"):
                with contextlib.suppress(OSError):
                    stale.unlink()
        return stats

    # ── 导出 / 备份 ─────────────────────────────────────────────────
    def prune_exports(self, *, keep: int, ttl_days: int) -> int:
        if not self.exports_dir.is_dir():
            return 0
        files = sorted(
            self.exports_dir.glob("*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        now, removed = _now(), 0
        for idx, path in enumerate(files):
            too_old = ttl_days > 0 and now - path.stat().st_mtime > ttl_days * 86400
            if idx >= keep or too_old:
                with contextlib.suppress(OSError):
                    path.unlink()
                removed += 1
        return removed

    def backup(self, tag: str = "pre_import", mode: str = "replace") -> Path | None:
        """把当前记忆打成可再次导入的 zip（含 manifest），存进 backups/。"""
        data = self.load_all()
        if not data:
            return None
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        name = f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}_{mode}.zip"
        dest = self.backups_dir / name
        tmp = self.tmp_dir / name
        now = _now()
        users = []
        for uid in sorted(data):
            items = data[uid]
            users.append(
                {
                    "uid": uid,
                    "items": len(items),
                    "updated_at": max(int(m.get("updated_at", 0)) for m in items),
                }
            )
        manifest = {
            "format": ARCHIVE_FORMAT,
            "format_version": 1,
            "schema_version": SCHEMA_VERSION,
            "plugin": "astrbot_plugin_long_memory",
            "plugin_version": "",
            "exported_at": now,
            "exported_by": "",
            "partial": False,
            "counts": {"users": len(users), "items": sum(u["items"] for u in users)},
            "users": users,
        }
        stamp = time.localtime()[:6]
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            # manifest 必须有，否则这份备份自己都导不回去
            zf.writestr(
                zipfile.ZipInfo("manifest.json", stamp),
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )
            for uid in sorted(data):
                zf.writestr(
                    zipfile.ZipInfo(f"memory/{uid}.json", stamp),
                    json.dumps(
                        {
                            "v": SCHEMA_VERSION,
                            "uid": uid,
                            "items": data[uid],
                            "updated_at": now,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
        os.replace(tmp, dest)
        return dest


# ── 插件状态 ─────────────────────────────────────────────────────────
class StateStore:
    """state.json：记录导入过的包指纹。写坏不致命，读失败一律当空。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.path = self.root / "state.json"
        self.tmp_dir = self.root / ".tmp"

    def load(self) -> dict[str, Any]:
        try:
            obj = json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return {}
        return obj if isinstance(obj, dict) else {}

    def save(self, obj: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            self.path, json.dumps(obj, ensure_ascii=False, indent=2), self.tmp_dir
        )

    def patch(self, **kwargs: Any) -> dict[str, Any]:
        state = self.load()
        state.update(kwargs)
        self.save(state)
        return state
