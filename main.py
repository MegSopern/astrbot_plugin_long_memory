"""
长期记忆插件入口（文件存储版）。

钩子做的两件事：
1. 请求前把记忆拼成 `<user_memory>` 块，append 到 extra_user_content_parts（mark_as_temp）
2. 从本轮用户消息里按规则抽偏好，异步写进插件数据目录（不阻塞回复）

"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools

try:  # AstrBot 用 data.plugins.<name>.main 导入，走相对导入
    from .memory_core import (
        append_block,
        build_injection,
        cfg_bool,
        cfg_int,
        extract_candidates,
        merge_import,
        record,
        sort_for_display,
    )
    from .store import MemoryStore, StateStore, sha256_of
    from .transfer import ArchiveError, build_archive, read_archive
except ImportError:  # 被当作顶层模块导入时（单测）
    from memory_core import (  # type: ignore[no-redef]
        append_block,
        build_injection,
        cfg_bool,
        cfg_int,
        extract_candidates,
        merge_import,
        record,
        sort_for_display,
    )
    from store import MemoryStore, StateStore, sha256_of  # type: ignore[no-redef]
    from transfer import (  # type: ignore[no-redef]
        ArchiveError,
        build_archive,
        read_archive,
    )

try:  # 4.24+ 才有；缺失时整体降级为"不注入"，不影响回复
    from astrbot.core.agent.message import TextPart
except Exception:  # pragma: no cover
    TextPart = None  # type: ignore[assignment]

try:  # @ 段类型：老版本没有该模块时退化为空元组，isinstance 恒为 False
    from astrbot.api.message_components import At as _AtSegment

    _AT_TYPES: tuple[type, ...] = (_AtSegment,)
except Exception:  # pragma: no cover
    _AT_TYPES = ()

TAG = "[long_memory]"
PLUGIN_VERSION = "0.4.0"
DEDUPE_WINDOW = 60  # 同一句话 60 秒内只抽一次
DISPLAY_LIMIT = 30  # /记忆 最多列几行
INJECT_TIMEOUT = 0.3  # 注入读盘 + 排序的硬上限（秒）
IMPORT_HISTORY_LIMIT = 20


def _uid_of(event: AstrMessageEvent) -> str:
    """只认 QQ 号：webchat 的 openid、cron 的假 id 一律不记。"""
    try:
        sid = str(event.get_sender_id() or "").strip()
    except Exception:
        return ""
    return sid if (sid.isdigit() and 5 <= len(sid) <= 12) else ""


class Main(Star):
    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.cfg: dict = config if isinstance(config, dict) else {}
        self.data_dir: Path | None = None
        self.store: MemoryStore | None = None
        self.state: StateStore | None = None
        self._tasks: set[asyncio.Task] = set()
        self._seen: dict[str, tuple[str, float]] = {}
        self._warned_no_textpart = False

    # ── 生命周期 ────────────────────────────────────────────────────
    async def initialize(self) -> None:
        """任何异常都不能外抛：AstrBot 不包 try，抛了插件就加载失败。"""
        try:
            self.data_dir = StarTools.get_data_dir(getattr(self, "name", "") or None)
            self.store = MemoryStore(self.data_dir)
            self.store.prepare()
            self.state = StateStore(self.data_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "%s 初始化失败，记忆功能停用：%s: %s", TAG, type(exc).__name__, exc
            )
            self.store = None
            return

        try:
            self.store.prune_exports(
                keep=cfg_int(self.cfg, "export_keep", 5, 1, 50),
                ttl_days=cfg_int(self.cfg, "export_ttl_days", 7, 0, 90),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s 清理导出目录失败：%s", TAG, exc)

        logger.info(
            "%s loaded v%s (extract=%s inject=%s max_items=%s dir=%s)",
            TAG,
            PLUGIN_VERSION,
            cfg_bool(self.cfg, "extract_enabled", True),
            cfg_bool(self.cfg, "inject_enabled", True),
            cfg_int(self.cfg, "max_items", 30, 5, 100),
            self.data_dir,
        )

    # ── 钩子 ────────────────────────────────────────────────────────
    @filter.on_llm_request()
    async def on_llm_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """注意：必须是 async def 且不能 yield。两段各自兜异常，互不牵连。"""
        if not cfg_bool(self.cfg, "enable", True) or self.store is None:
            return
        uid = _uid_of(event)
        if not uid:
            return

        if cfg_bool(self.cfg, "inject_enabled", True):
            try:
                await self._inject(uid, req)
            except asyncio.TimeoutError:
                logger.warning("%s inject timeout, skipped this turn", TAG)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s inject failed: %s: %s", TAG, type(exc).__name__, exc)

        if cfg_bool(self.cfg, "extract_enabled", True):
            try:
                self._schedule_extract(uid, event)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "%s schedule failed: %s: %s", TAG, type(exc).__name__, exc
                )

    async def _inject(self, uid: str, req: ProviderRequest) -> None:
        if TextPart is None:
            if not self._warned_no_textpart:
                self._warned_no_textpart = True
                logger.warning(
                    "%s TextPart unavailable, memory injection disabled", TAG
                )
            return
        block = await asyncio.wait_for(
            build_injection(self.store, uid, self.cfg, query=self._query_of(req)),
            timeout=INJECT_TIMEOUT,
        )
        if not block:
            return
        parts = getattr(req, "extra_user_content_parts", None)
        append_block(parts, block, TextPart)

    @staticmethod
    def _query_of(req: ProviderRequest) -> str:
        """本轮用户输入。ProviderRequest() 默认 prompt 是 None，不能直接当字符串用。"""
        raw = getattr(req, "prompt", None)
        return raw if isinstance(raw, str) else ""

    def _schedule_extract(self, uid: str, event: AstrMessageEvent) -> None:
        text = (event.message_str or "").strip()
        if len(text) < 4 or text.startswith("/"):
            return
        # 同轮去重：工具循环 / 重试会让钩子对同一句话多次触发，否则 hits 会虚涨
        now = time.monotonic()
        mark = f"{len(text)}:{text[:40]}"
        prev = self._seen.get(uid)
        if prev and prev[0] == mark and now - prev[1] < DEDUPE_WINDOW:
            return
        self._seen[uid] = (mark, now)

        candidates = extract_candidates(text)
        if not candidates:
            return
        task = asyncio.create_task(record(self.store, uid, candidates, self.cfg))
        self._tasks.add(task)
        task.add_done_callback(self._on_extract_done)

    def _on_extract_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:  # 只记类型，不记正文，避免隐私进日志
            logger.warning("%s extract failed: %s: %s", TAG, type(exc).__name__, exc)

    # ── 查询 / 清除 ─────────────────────────────────────────────────
    # 权限在方法内判：不带 QQ 谁都能操作自己；带了 QQ 只有超级管理员能指向他人。
    @filter.command("记忆")
    async def memory_cmd(self, event: AstrMessageEvent, qq: str = ""):
        """查看长期记忆。用法：/记忆、/记忆 <QQ号>、/记忆 @某人（后两者仅超级管理员）"""
        if not self._cmd_allowed(event):
            yield event.plain_result("这个指令请在私聊里用。")
            return
        if self.store is None:
            yield event.plain_result("记忆存储没初始化起来，先看日志。")
            return
        target, err = self._resolve_target(event, qq, "记忆")
        if err:
            yield event.plain_result(err)
            return
        items = self.store.load_sync(target)
        if not items:
            yield event.plain_result(f"{target} 还没有记忆。")
            return
        ordered = sort_for_display(items)
        shown = ordered[:DISPLAY_LIMIT]
        lines = []
        for idx, mem in enumerate(shown, 1):
            suffix = f" ×{mem['hits']}" if mem.get("hits", 1) > 1 else ""
            lines.append(f"{idx}. {mem['text']}{suffix}")
        total = len(items)
        if total <= DISPLAY_LIMIT:
            head = f"长期记忆（{target} · {total} 条）"
        else:
            head = f"长期记忆（{target} · 共 {total} 条，仅显示前 {DISPLAY_LIMIT} 条）"
        yield event.plain_result(head + "\n" + "\n".join(lines))

    @filter.command("忘记")
    async def forget_cmd(self, event: AstrMessageEvent, qq: str = ""):
        """清除某个人的长期记忆。用法：/忘记、/忘记 <QQ号>、/忘记 @某人（后两者仅超级管理员）"""
        if not self._cmd_allowed(event):
            yield event.plain_result("这个指令请在私聊里用。")
            return
        if self.store is None:
            yield event.plain_result("记忆存储没初始化起来，先看日志。")
            return
        target, err = self._resolve_target(event, qq, "忘记")
        if err:
            yield event.plain_result(err)
            return
        count = await self.store.clear(target)
        yield event.plain_result(f"已清除 {count} 条记忆。")

    # ── 导出 / 导入 / 清空 ──────────────────────────────────────────
    @filter.command("导出")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def export_cmd(self, event: AstrMessageEvent):
        """把全部记忆打包成 zip 并给出下载链接。"""
        if not self._cmd_allowed(event):
            yield event.plain_result("导出请在私聊里用，链接不能发到群里。")
            return
        if self.store is None:
            yield event.plain_result("记忆存储没初始化起来，先看日志。")
            return

        async with self.store.bulk:
            data = self.store.load_all()
            if not data:
                yield event.plain_result("还没有任何记忆，不用导出。")
                return
            name = f"long_memory_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.zip"
            dest = self.store.exports_dir / name
            try:
                manifest = await asyncio.to_thread(
                    build_archive,
                    dest,
                    data,
                    plugin_version=PLUGIN_VERSION,
                    exported_by=_uid_of(event),
                    tmp_dir=self.store.tmp_dir,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s export failed: %s", TAG, exc)
                yield event.plain_result(f"导出失败：{type(exc).__name__}: {exc}")
                return
            try:
                self.store.prune_exports(
                    keep=cfg_int(self.cfg, "export_keep", 5, 1, 50),
                    ttl_days=cfg_int(self.cfg, "export_ttl_days", 7, 0, 90),
                )
            except Exception:  # noqa: BLE001
                pass

        counts = manifest.get("counts") or {}
        link = await self._make_link(dest)
        hint = "1 小时内有效；拿到链接的人无需登录即可下载，别外传"
        body = (
            f"已导出 {counts.get('users', 0)} 人 / {counts.get('items', 0)} 条\n{name}"
        )
        if link:
            body += f"\n{hint}：\n{link}"
        else:
            body += (
                f"\n\n下载：把 data/plugin_data/{getattr(self, 'name', '')}"
                f"/exports/{name} 拖进浏览器，或配置 panel_base_url 后重新导出拿链接。"
            )
        yield event.plain_result(body)

    @filter.command("导入")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def import_cmd(self, event: AstrMessageEvent, mode: str = "", hint: str = ""):
        """从 WebUI 上传的 zip 导入记忆。用法：/导入 [覆盖|合并] [文件名片段]"""
        if not self._cmd_allowed(event):
            yield event.plain_result("导入请在私聊里用。")
            return
        if self.store is None or self.state is None:
            yield event.plain_result("记忆存储没初始化起来，先看日志。")
            return

        mode_key = self._resolve_mode(mode)
        if mode_key is None:
            yield event.plain_result("用法：/导入 覆盖 或 /导入 合并")
            return

        picked, skipped, total = self._pick_archive(hint)
        if picked is None:
            if total and skipped == total:
                yield event.plain_result(
                    f"找到 {total} 个包，但都导入过了。要重导请重新上传一个包。"
                )
            else:
                yield event.plain_result(
                    "没找到上传的 zip。请到 WebUI → 插件 → 长期记忆 → 配置 →「导入包（zip）」"
                    "上传后点保存，再执行 /导入。"
                )
            return
        path = picked

        try:
            sha = await asyncio.to_thread(sha256_of, path)
        except OSError as exc:
            yield event.plain_result(f"读不了这个包：{exc}")
            return
        imported = dict((self.state.load().get("imported") or {}))
        if sha in imported:
            yield event.plain_result(
                f"{path.name} 已经导入过了（{imported[sha].get('mode')}）。"
                "要重导请重新上传一个包。"
            )
            return

        try:
            manifest, data, stats = await asyncio.to_thread(
                read_archive,
                path,
                max_mb=cfg_int(self.cfg, "import_max_mb", 32, 1, 200),
            )
        except ArchiveError as exc:
            yield event.plain_result(f"包不合法：{exc}")
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s import read failed: %s", TAG, exc)
            yield event.plain_result(f"读取失败：{type(exc).__name__}")
            return

        if not data:
            yield event.plain_result("包里没有任何记忆，已中止（不会清空现有数据）。")
            return

        backup_path = None
        if cfg_bool(self.cfg, "import_backup", True):
            try:
                backup_path = await asyncio.to_thread(
                    self.store.backup, "pre_import", mode_key
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s backup failed: %s", TAG, exc)
                yield event.plain_result(
                    f"备份当前数据失败（{type(exc).__name__}），已中止导入。"
                )
                return

        async with self.store.bulk:
            if mode_key == "replace":
                delete_absent = not bool(manifest.get("partial", False))
                written, deleted = await self.store.write_many(
                    data, delete_absent=delete_absent
                )
                if delete_absent:
                    detail = f"写入 {written} 人；删除本地 {deleted} 人（不在包内）"
                else:
                    detail = f"写入 {written} 人（部分包，未删除包外用户）"
            else:
                added = updated = unchanged = dropped = 0
                for uid, incoming in data.items():
                    local = self.store.load_sync(uid)
                    merged, st = merge_import(local, incoming)
                    self.store.save_sync(uid, merged)
                    added += st["added"]
                    updated += st["updated"]
                    unchanged += st["unchanged"]
                    dropped += st["dropped"]
                detail = (
                    f"涉及 {len(data)} 人；新增 {added} 条，更新 {updated} 条，"
                    f"删除对立项 {dropped} 条，无变化 {unchanged} 条"
                )

        imported[sha] = {"name": path.name, "mode": mode_key, "at": int(time.time())}
        if len(imported) > IMPORT_HISTORY_LIMIT:
            for key in sorted(imported, key=lambda k: imported[k].get("at", 0))[
                :-IMPORT_HISTORY_LIMIT
            ]:
                imported.pop(key, None)
        try:
            self.state.patch(imported=imported, last_import=imported[sha])
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s 记录导入指纹失败：%s", TAG, exc)

        counts = manifest.get("counts") or {}
        lines = [
            f"包：{path.name}（{counts.get('users', 0)} 人 / {counts.get('items', 0)} 条）",
            f"模式：{'覆盖' if mode_key == 'replace' else '合并'}",
        ]
        if backup_path:
            lines.append(f"已备份当前数据 → {backup_path.name}")
        lines.append(f"结果：{detail}")
        if stats.get("dropped_items"):
            lines.append(f"丢弃无效条目 {stats['dropped_items']} 条")
        if skipped:
            lines.append(f"（跳过 {skipped} 个已导入过的包）")
        yield event.plain_result("\n".join(lines))

    @filter.command("清空全部")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def purge_cmd(self, event: AstrMessageEvent, arg: str = ""):
        """清空所有用户的记忆。用法：/清空全部 确认"""
        if not self._cmd_allowed(event):
            yield event.plain_result("这个指令请在私聊里用。")
            return
        if self.store is None:
            yield event.plain_result("记忆存储没初始化起来，先看日志。")
            return
        if arg.strip() not in ("确认", "confirm"):
            users = len(self.store.list_uids())
            yield event.plain_result(
                f"这会删除全部 {users} 人的记忆、导出包和备份，不可撤销。"
                "确认请发 /清空全部 确认"
            )
            return
        stats = await self.store.clear_all()
        yield event.plain_result(
            f"已删除 {stats['users']} 人的记忆、{stats['exports']} 个导出包、"
            f"{stats['backups']} 个备份。此操作不可撤销。"
        )

    # ── 辅助 ────────────────────────────────────────────────────────
    def _resolve_mode(self, mode: str) -> str | None:
        raw = (mode or "").strip().lower()
        if not raw:
            default = (
                str(self.cfg.get("import_default_mode") or "replace").strip().lower()
            )
            return "merge" if default == "merge" else "replace"
        if raw in ("覆盖", "replace", "overwrite"):
            return "replace"
        if raw in ("合并", "merge"):
            return "merge"
        return None

    def _pick_archive(self, hint: str = "") -> tuple[Path | None, int, int]:
        """返回 (选中的包, 跳过的已导入数, 候选总数)。候选 = 配置登记的 + 目录扫到的。"""
        if self.store is None or self.data_dir is None or self.state is None:
            return None, 0, 0
        candidates: list[Path] = []
        raw = self.cfg.get("import_zip") or []
        if isinstance(raw, list):
            for item in raw:
                rel = str(item).replace("\\", "/").lstrip("/")
                if ".." in rel.split("/"):
                    continue
                path = self.data_dir / rel
                if path.is_file():
                    candidates.append(path)
        upload_dir = self.data_dir / "files" / "import_zip"
        if upload_dir.is_dir():
            candidates.extend(upload_dir.glob("*.zip"))

        uniq: dict[Path, Path] = {}
        for path in candidates:
            if path.suffix.lower() != ".zip":
                continue
            try:
                uniq[path.resolve()] = path
            except OSError:
                continue
        ordered = sorted(uniq.values(), key=lambda p: p.stat().st_mtime, reverse=True)
        if hint:
            ordered = [p for p in ordered if hint in p.name]

        already = self.state.load().get("imported") or {}
        skipped = 0
        total = len(ordered)
        for path in ordered:
            try:
                if sha256_of(path) in already:
                    skipped += 1
                    continue
            except OSError:
                continue
            return path, skipped, total
        return None, skipped, total

    async def _make_link(self, path: Path) -> str:
        try:
            from astrbot.core import file_token_service

            token = await file_token_service.register_file(
                str(path),
                timeout=cfg_int(self.cfg, "export_token_ttl", 3600, 300, 86400),
                single_use=False,  # 默认是 True，一次访问就废，必须显式关掉
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s register_file failed: %s", TAG, exc)
            return ""
        base = str(self.cfg.get("panel_base_url") or "").strip().rstrip("/")
        if base and not base.lower().startswith(("http://", "https://")):
            base = ""
        return f"{base}/api/file/{token}" if base else f"/api/file/{token}"

    def _cmd_allowed(self, event: AstrMessageEvent) -> bool:
        try:
            in_group = bool(event.get_group_id())
        except Exception:
            in_group = False
        return (not in_group) or cfg_bool(self.cfg, "allow_group_cmd", False)

    def _resolve_target(
        self, event: AstrMessageEvent, qq: str, cmd: str
    ) -> tuple[str, str]:
        """解析指令目标，返回 (uid, 错误提示)；错误提示非空时调用方应中止。

        目标来源优先级：@ 用户 > 纯数字参数 > 自己。@ 必须从消息段取，
        因为文本里的「@昵称」反查不到 QQ 号。
        不带目标 → 操作自己，所有人可用；指向他人 → 只有超级管理员可以。
        「超级管理员」= event.is_admin()，即 AstrBot 配置里的 admins_id 全局管理员，
        与 PermissionType.ADMIN 判据同源。
        """
        me = _uid_of(event)
        arg = (qq or "").strip()
        usage = f"用法：/{cmd} 或 /{cmd} <QQ号>"

        at_ids: list[str] = []
        try:
            self_id = str(event.get_self_id() or "")
            for seg in event.get_messages() or []:
                if isinstance(seg, _AT_TYPES):
                    uid = str(getattr(seg, "qq", "") or "")
                    if uid.isdigit() and uid != self_id:  # @全体成员 / @机器人 不算目标
                        at_ids.append(uid)
        except Exception:  # 平台事件实现不全时按"没 @ 人"处理
            at_ids = []
        if len(at_ids) > 1:
            return "", f"一次只能指定一个人，这条消息 @ 了 {len(at_ids)} 个。"

        if at_ids:
            target = at_ids[0]
        elif arg.isdigit():
            target = arg
        elif not arg:
            return (me, "") if me else ("", usage)
        else:
            return "", usage

        if target == me:  # 指向自己，等同不带参数
            return target, ""
        try:
            allowed = bool(event.is_admin())
        except Exception:  # 平台事件实现不全时按无权限处理
            allowed = False
        if allowed:
            return target, ""
        return (
            "",
            f"只能操作自己的记忆；要看或清别人的，请让超级管理员执行 /{cmd} <QQ号>。",
        )

    async def terminate(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
