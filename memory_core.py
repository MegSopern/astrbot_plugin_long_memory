"""长期记忆的纯逻辑层。

这里刻意不 import astrbot：抽取规则、去重合并、注入块生成全部是纯函数，
配 store 对象调用即可，方便脱离框架跑单测。
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

# ── 常量 ──────────────────────────────────────────────────────────────
MAX_OBJ_LEN = 24          # 宾语最大字数
MAX_TEXT_LEN = 34         # 存储文本最大字数
MAX_INPUT = 400           # 输入截断，防正则回溯
MAX_CLAUSES = 6           # 单条消息最多看几个分句
MAX_PER_MESSAGE = 2       # 单条消息最多抽几条
PIN_HITS = 3              # 命中这么多次视为"常驻"，不再过期
STICKY_KINDS = frozenset({"name", "city", "job"})
OPPOSITE = {"like": "dislike", "dislike": "like"}

# 注入预算：默认值必须与 _conf_schema.json 的 default 一致（有测试盯着）
INJECT_MAX_ITEMS_DEFAULT = 30
INJECT_MAX_ITEMS_RANGE = (1, 50)
INJECT_MAX_CHARS_DEFAULT = 500
INJECT_MAX_CHARS_RANGE = (40, 800)

# ── 隐私：命中任一条就整条不记 ─────────────────────────────────────────
_PII_PATTERNS = (
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),          # 手机号
    re.compile(r"\d{17}[\dXx]"),                        # 身份证
    re.compile(r"\d{4,}"),                              # QQ/银行卡/门牌/验证码/工号
    re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}"),       # 邮箱
    re.compile(r"(?i)(?:https?://|www\.)\S+"),          # 链接
)

_SENSITIVE_WORDS = (
    "身份证", "护照", "银行卡", "密码", "验证码", "手机号", "电话", "地址",
    "住址", "门牌", "工号", "学号", "病历", "手术", "抑郁", "怀孕", "工资",
    "薪水", "生日", "定位", "坐标", "车牌",
)

# 句首是别人 → 不是说话人自己的偏好
_OTHER_PEOPLE = re.compile(
    r"^(?:他|她|它|他们|她们|别人|"
    r"我(?:妈|爸|朋友|同事|同学|对象|男朋友|女朋友|老公|老婆|孩子|儿子|女儿|哥|姐|弟|妹))"
)

# 显式撤回：对整条消息先生效，再分句
_OPT_OUT = re.compile(r"别记|不要记|不用记|别存|忘掉这|别记住")

_QUESTION = re.compile(r"[?？]|吗|呢|什么|怎么|为什么|哪|谁|多少|几点|是不是")
_NESTED_NEG = re.compile(r"不是不|并非不|哪有|才不|根本不|从来不|没有不")
_REPORTED = re.compile(r"^(?:他|她|它|他们|她们|别人|有人说|听说|据说)")

# ── 宾语过滤：稳定偏好 vs 临时情绪 ────────────────────────────────────
_BAD_OBJECTS = {
    "你", "您", "他", "她", "它", "你们", "他们", "她们", "大家", "别人", "谁",
    "这样", "那样", "这个", "那个", "这种", "那种", "这么", "那么", "这些", "那些",
    "这里", "那里", "这儿", "那儿", "如此", "这样吧", "这种感觉", "那种感觉",
    "今天", "现在", "刚才", "这次", "最近", "这两天", "今晚", "明天", "这个周末",
}
_BAD_OBJECT_PREFIX_ALWAYS = ("被", "让", "给", "使", "你", "您", "他", "她")
_BAD_OBJECT_PREFIX_PHRASES = (
    "这样", "那样", "这么", "那么", "这个", "那个", "这种", "那种", "这些", "那些",
    "这里", "那里", "这次", "那次", "今天", "现在", "刚才", "随便", "啥", "什么",
)
_BAD_OBJECT_SUFFIX = ("了", "吗", "呢", "吧", "啊", "呀", "哦", "啦", "嘛", "么", "的话")
_ONE_CHAR_OK = {"辣", "甜", "苦", "酸", "咸", "麻"}
_LEAD_VERBS = ("喝", "吃", "穿", "看", "玩", "用", "听", "读", "闻", "抽")
_NON_PLACE = {
    "本地", "外地", "外国", "正常", "普通", "一般", "好人", "坏人",
    "男人", "女人", "大人", "小人", "众人", "路人",
}

# ── 抽取规则 ─────────────────────────────────────────────────────────
_MOD = r"(?:很|超|特别|非常|挺|比较|最|真的|蛮|有点儿?|有一点)?"
_WHEN = r"(?:现在|最近|一直|平时|一般|通常)?"


class Rule:
    __slots__ = ("kind", "pattern", "render")

    def __init__(self, kind: str, pattern: re.Pattern[str], render: str) -> None:
        self.kind = kind
        self.pattern = pattern
        self.render = render


_RULES: tuple[Rule, ...] = (
    Rule(
        "like",
        re.compile(
            rf"^我?{_WHEN}{_MOD}"
            rf"(?:喜欢|爱|中意|偏爱|迷上)(?:上|了)?(?P<obj>.{{1,{MAX_OBJ_LEN}}})$"
        ),
        "verbatim",
    ),
    Rule(
        "dislike",
        re.compile(
            rf"^我?{_WHEN}{_MOD}"
            rf"(?:不喜欢|不太喜欢|不爱|不太爱|讨厌|受不了|反感|厌恶|吃不消|接受不了)"
            rf"(?:了)?(?P<obj>.{{1,{MAX_OBJ_LEN}}})$"
        ),
        "verbatim",
    ),
    Rule(
        "name",
        re.compile(
            r"^(?:我(?:的名字)?叫|我的名字是|你可以叫我|叫我)"
            r"(?P<obj>[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_]{0,11})$"
        ),
        "name",
    ),
    Rule(
        "city",
        re.compile(r"^我(?:现在)?(?:住在|家住|家在|定居在)(?P<obj>[\u4e00-\u9fff]{2,12})$"),
        "city",
    ),
    Rule(
        "city",
        re.compile(r"^我是(?P<obj>[\u4e00-\u9fff]{2,6})人$"),
        "city",
    ),
    Rule(
        "job",
        re.compile(
            r"^我(?:现在)?在(?P<obj>[\u4e00-\u9fffA-Za-z0-9]{2,15}?)(?:工作|上班|打工|实习)$"
        ),
        "job",
    ),
    Rule(
        "habit",
        re.compile(
            r"^我(?P<trig>每天|每晚|每早|每周|每月|平时|一般|通常|习惯|总是|经常)"
            r"(?P<obj>.{2,18})$"
        ),
        "habit",
    ),
)


# ── 工具函数 ─────────────────────────────────────────────────────────
def cfg_int(cfg: dict[str, Any] | None, key: str, default: int, lo: int, hi: int) -> int:
    try:
        val = int((cfg or {}).get(key, default))
    except (TypeError, ValueError):
        val = default
    return max(lo, min(hi, val))


def cfg_bool(cfg: dict[str, Any] | None, key: str, default: bool) -> bool:
    val = (cfg or {}).get(key, default)
    return val if isinstance(val, bool) else default


def _object_ok(obj: str) -> bool:
    o = obj.strip().strip("，,。.！!？?、~～ ")
    if not o or len(o) > MAX_OBJ_LEN:
        return False
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", o):
        return False
    if o in _BAD_OBJECTS or o in _NON_PLACE:
        return False
    if len(o) == 1 and o not in _ONE_CHAR_OK:
        return False
    if o.startswith(_BAD_OBJECT_PREFIX_ALWAYS):
        return False
    if o.startswith(_BAD_OBJECT_PREFIX_PHRASES):
        return False
    if len(o) <= 3 and o[0] in "这那":  # 这杯/那样：指代，不是具体偏好
        return False
    if o.endswith(_BAD_OBJECT_SUFFIX):
        return False
    if _OTHER_PEOPLE.match(o):
        return False
    return True


def _privacy_ok(clause: str, obj: str) -> bool:
    """宁可漏记，不可误记：任一命中就整条放弃。"""
    blob = f"{clause} {obj}"
    if _OPT_OUT.search(blob):
        return False
    if any(kw in blob for kw in _SENSITIVE_WORDS):
        return False
    return not any(p.search(blob) for p in _PII_PATTERNS)


def _norm_obj(obj: str) -> str:
    """归一化宾语，用于判断"是不是同一件事"。"""
    o = re.sub(r"\s+", "", obj)
    o = re.sub(r"^(?:很|超|特别|非常|挺|比较|最|真的|蛮|有点儿?|有一点)+", "", o)
    o = o.rstrip("的")
    for verb in _LEAD_VERBS:
        if o.startswith(verb) and len(o) - len(verb) >= 1:
            o = o[len(verb):]
            break
    o = o.replace("喜欢", "").replace("讨厌", "")
    return o.lower()


def _render(rule: Rule, clause: str, obj: str, match: re.Match[str]) -> str:
    if rule.render == "name":
        text = f"称呼：{obj}"
    elif rule.render == "city":
        text = f"住在{obj}"
    elif rule.render == "job":
        text = f"在{obj}工作"
    elif rule.render == "habit":
        text = f"{match.group('trig')}{obj}"
    else:
        text = re.sub(r"^我", "", clause).strip()
    return text[:MAX_TEXT_LEN]


def split_clauses(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    parts = re.split(r"[，,。．.!！?？;；~～\n\r]+", text)
    return [p.strip() for p in parts if p.strip()]


def _match_clause(clause: str) -> dict[str, Any] | None:
    if not (4 <= len(clause) <= 60):
        return None
    if _QUESTION.search(clause) or _NESTED_NEG.search(clause) or _REPORTED.match(clause):
        return None
    if _OTHER_PEOPLE.match(clause):
        return None

    for rule in _RULES:
        match = rule.pattern.match(clause)
        if not match:
            continue
        obj = match.group("obj").strip().strip("，,。.！!？?、~～ ")
        # 句式命中但内容不合格：本条直接放弃，不回退到更弱的规则
        if not _object_ok(obj) or not _privacy_ok(clause, obj):
            return None
        nobj = _norm_obj(obj)
        if not nobj:
            return None
        now = int(time.time())
        return {
            "id": uuid.uuid4().hex[:12],
            "kind": rule.kind,
            "text": _render(rule, clause, obj, match),
            "nobj": nobj,
            "hits": 1,
            "created_at": now,
            "updated_at": now,
        }
    return None


def extract_candidates(text: str) -> list[dict[str, Any]]:
    """从一条用户消息里最多抽 MAX_PER_MESSAGE 条候选记忆。"""
    if not text:
        return []
    text = text[:MAX_INPUT]
    if _OPT_OUT.search(text):
        return []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for clause in split_clauses(text)[:MAX_CLAUSES]:
        cand = _match_clause(clause)
        if cand is None or cand["nobj"] in seen:
            continue
        seen.add(cand["nobj"])
        out.append(cand)
        if len(out) >= MAX_PER_MESSAGE:
            break
    return out


# ── 合并 / 淘汰 ──────────────────────────────────────────────────────
def _evict_key(item: dict[str, Any]) -> tuple:
    """排序键，越小越先被淘汰：非粘性 → 未置顶 → 命中少 → 更旧。"""
    return (
        item.get("kind") in STICKY_KINDS,
        int(item.get("hits", 1)) >= PIN_HITS,
        int(item.get("hits", 1)),
        int(item.get("updated_at", 0)),
    )


def _find_same(items: list[dict[str, Any]], cand: dict[str, Any]) -> dict[str, Any] | None:
    for item in items:
        if item.get("kind") != cand["kind"]:
            continue
        if item.get("nobj") == cand["nobj"]:
            return item
        old, new = str(item.get("nobj", "")), cand["nobj"]
        if len(old) >= 2 and len(new) >= 2 and (old in new or new in old):
            return item
    return None


def merge_items(
    existing: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    max_items: int = 30,
    ttl_days: int = 180,
    now: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """返回 (合并后的全量, 本轮有变化的条目)。"""
    now = int(time.time()) if now is None else int(now)
    items = [dict(m) for m in existing if isinstance(m, dict) and m.get("text")]
    changed: list[dict[str, Any]] = []

    for cand in candidates:
        opposite = OPPOSITE.get(cand["kind"])
        if opposite:  # 改口：删掉态度相反的同宾语旧条目
            items = [
                m for m in items
                if not (m.get("kind") == opposite and m.get("nobj") == cand["nobj"])
            ]

        hit = _find_same(items, cand)
        if hit is None:
            items.append(dict(cand))
            changed.append(cand)
            continue

        hit["hits"] = int(hit.get("hits", 1)) + 1
        hit["updated_at"] = now
        # 新说法更具体时（旧宾语是新宾语的子串），用新的替换旧的
        if len(cand["text"]) > len(hit.get("text", "")) and str(hit.get("nobj", "")) in cand["nobj"]:
            hit["text"], hit["nobj"] = cand["text"], cand["nobj"]
        changed.append(hit)

    if ttl_days > 0:
        cutoff = now - ttl_days * 86400
        items = [
            m for m in items
            if m.get("kind") in STICKY_KINDS
            or int(m.get("hits", 1)) >= PIN_HITS
            or int(m.get("updated_at", 0)) >= cutoff
        ]

    limit = max(1, max_items)
    while len(items) > limit:  # 硬上限优先：哪怕全是粘性也要收敛
        items.remove(min(items, key=_evict_key))

    return items, changed


# ── 导入合并（与 merge_items 分开：导入不是"又说了一遍"，不许累加 hits） ──
def _match_for_import(items: list[dict[str, Any]], cand: dict[str, Any]) -> dict[str, Any] | None:
    cid = cand.get("id")
    if isinstance(cid, str) and cid:
        for item in items:
            if item.get("id") == cid:
                return item
    for item in items:
        if item.get("kind") == cand["kind"] and item.get("nobj") == cand["nobj"]:
            return item
    for item in items:
        if item.get("kind") != cand["kind"]:
            continue
        old, new = str(item.get("nobj", "")), cand["nobj"]
        if len(old) >= 2 and len(new) >= 2 and (old in new or new in old):
            return item
    return None


def merge_import(
    local: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    per_uid_cap: int = 500,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """导入合并：幂等、hits 取较大值、不累加，改口时删掉对立项。

    对同一个包连续跑两次，结果必须完全一致（导入 ≠ 用户又说了一遍）。
    """
    items = [dict(m) for m in local if isinstance(m, dict) and m.get("text")]
    stats = {"added": 0, "updated": 0, "unchanged": 0, "dropped": 0, "truncated": 0}

    for raw in incoming or []:
        if not isinstance(raw, dict) or not raw.get("text"):
            continue
        cand = dict(raw)
        incoming_updated = int(cand.get("updated_at", 0) or 0)

        opposite = OPPOSITE.get(cand.get("kind"))
        if opposite:
            for old in [
                m for m in items
                if m.get("kind") == opposite
                and m.get("nobj") == cand.get("nobj")
                and int(m.get("updated_at", 0) or 0) <= incoming_updated
            ]:
                items.remove(old)
                stats["dropped"] += 1

        hit = _match_for_import(items, cand)
        if hit is None:
            items.append(cand)
            stats["added"] += 1
            continue

        before = (
            hit.get("text"), hit.get("nobj"), int(hit.get("hits", 1)),
            int(hit.get("updated_at", 0)),
        )
        prev_updated = int(hit.get("updated_at", 0) or 0)
        hit["hits"] = max(int(hit.get("hits", 1)), int(cand.get("hits", 1)))
        hit["created_at"] = min(int(hit.get("created_at", 0)), int(cand.get("created_at", 0)))
        hit["updated_at"] = max(prev_updated, incoming_updated)
        # 更晚、更具体的说法才覆盖旧文本
        if incoming_updated >= prev_updated and len(str(cand.get("text", ""))) > len(str(hit.get("text", ""))):
            hit["text"], hit["nobj"] = cand["text"], cand["nobj"]

        after = (
            hit.get("text"), hit.get("nobj"), int(hit.get("hits", 1)),
            int(hit.get("updated_at", 0)),
        )
        stats["updated" if after != before else "unchanged"] += 1

    cap = max(1, per_uid_cap)
    while len(items) > cap:
        items.remove(min(items, key=_evict_key))
        stats["truncated"] += 1

    return items, stats


# ── 注入块 ───────────────────────────────────────────────────────────
_BLOCK_HEADER = '<user_memory hint="参考用，可能过时；勿逐条复述；群聊勿主动提及">'
_BLOCK_FOOTER = "</user_memory>"
_BLOCK_TAG = "<user_memory"  # 只认标签名，不认 hint 文案（文案会改，去重不该跟着坏）

# 整块长度 = BLOCK_FIXED + Σ(len(text) + 3)，其中 3 是每行的 "- " 与换行。
# select_for_injection 与 build_block 必须用同一个 max_chars 口径，否则会出现
# 「选中了却被静默 pop 掉」。
BLOCK_FIXED = len(_BLOCK_HEADER) + len(_BLOCK_FOOTER) + 1
BLOCK_TEXT_BUDGET = INJECT_MAX_CHARS_DEFAULT
BLOCK_HARD_LIMIT = 900  # 绝对兜底：slider 上限 800 + 58，取整


# ── 本轮相关性（字符级，不依赖分词） ─────────────────────────────────
# 按长度降序替换，否则「不喜欢」会先被「喜欢」切成「不」。
# 必须剥掉「喜欢/讨厌」：它们出现在几乎每条记忆里，不剥就等于没有排序。
_REL_STOP_WORDS = (
    "不喜欢", "不太喜欢", "喜欢", "讨厌", "偏爱", "中意", "受不了", "反感",
    "什么", "怎么样", "怎么", "为什么", "哪个", "哪些", "哪里", "哪儿",
    "多少", "几点", "是不是", "有没有",
    "这个", "那个", "这些", "那些", "这样", "那样", "这种", "那种",
    "我们", "你们", "他们", "她们", "自己", "大家",
    "现在", "最近", "平时", "一般", "通常", "刚才", "今天", "明天",
    "知道", "记得", "告诉", "觉得", "可以", "一下", "还是", "或者",
    "因为", "所以", "但是", "如果",
)
_REL_STOP_WORDS_BY_LEN = tuple(sorted(_REL_STOP_WORDS, key=len, reverse=True))

_REL_STOP_CHARS = frozenset(
    "我你您他她它的了着过吗呢吧啊呀哦嘛"
    "是在有和与及就都也还真很太更最不没会要想能可让给把被对从到"
    "这那些个们么什怎哪几谁多少"
)


def query_chars(query: Any) -> set[str]:
    """把本轮用户输入压成「有信息量的字符集合」。空 / 非字符串 → 空集合。"""
    if not isinstance(query, str) or not query.strip():
        return set()
    text = query[:MAX_INPUT]  # 复用 400 截断，防超长输入
    for word in _REL_STOP_WORDS_BY_LEN:
        text = text.replace(word, " ")
    out: set[str] = set()
    for ch in text:
        if ch in _REL_STOP_CHARS:
            continue
        if ch.isalnum() or "\u4e00" <= ch <= "\u9fff":
            out.add(ch.lower())
    return out


def relevance_of(item: dict[str, Any], qchars: set[str]) -> int:
    """字符级重合数（0 表示无关）。

    同时看 nobj：text 被截到 34 字时，关键宾语可能只在 nobj 里完整。
    这是**字面**匹配：问「什么颜色」能蹭到「蓝色」（共用「色」字），
    但问「什么花」蹭不到「苹果」。真正的兜底是全量注入，不是这里。
    """
    if not qchars:
        return 0
    hay = f"{item.get('text') or ''} {item.get('nobj') or ''}".lower()
    if not hay.strip():
        return 0
    return sum(1 for ch in qchars if ch in hay)


def _int_field(item: dict[str, Any], key: str, default: int) -> int:
    value = item.get(key, default)
    if isinstance(value, bool):  # JSON 的 true 是 int 子类，必须排除
        return default
    try:
        got = int(value)
    except (TypeError, ValueError):
        return default
    return got if got >= 0 else default


def _rank_key(item: dict[str, Any], rel: int, idx: int) -> tuple:
    """粘性优先，其次本轮相关，再按常驻/命中/新旧，最后用原始下标兜底（结果完全确定）。

    为什么不把相关放第一位：存储上限可到 100 条而注入上限 30 条，
    若前 30 条都是弱相关，排最后的「住在杭州」会被 max_items 永久挤掉。
    """
    hits = _int_field(item, "hits", 1)
    return (
        0 if item.get("kind") in STICKY_KINDS else 1,
        0 if rel > 0 else 1,
        -rel,
        0 if hits >= PIN_HITS else 1,
        -hits,
        -_int_field(item, "updated_at", 0),
        idx,
    )


def sort_for_display(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`/记忆` 的展示顺序：hits 降序，同 hits 保持传入（文件）顺序。

    sorted 是稳定排序，只按 -hits 排就能保证同 hits 不打乱原序。
    """
    return sorted(items, key=lambda m: -_int_field(m, "hits", 1))


def select_for_injection(
    items: list[dict[str, Any]],
    *,
    query: Any = None,
    max_items: int = INJECT_MAX_ITEMS_DEFAULT,
    max_chars: int = INJECT_MAX_CHARS_DEFAULT,
) -> list[dict[str, Any]]:
    qchars = query_chars(query)
    scored = [(item, idx, relevance_of(item, qchars)) for idx, item in enumerate(items)]
    scored.sort(key=lambda row: _rank_key(row[0], row[2], row[1]))

    picked: list[dict[str, Any]] = []
    used = 0
    for item, _idx, _rel in scored:
        if len(picked) >= max_items:
            break
        text = str(item.get("text") or "")
        if not text:
            continue
        cost = len(text) + 3  # "- " + 换行
        if used + cost > max_chars:
            continue  # 刻意用 continue：让后面的短条目填满预算缝隙
        picked.append(item)
        used += cost
    return picked


def build_block(picked: list[dict[str, Any]], *, max_chars: int | None = None) -> str:
    """max_chars 与 select_for_injection 同口径：只算正文 + 每行 3 字符。"""
    budget = BLOCK_TEXT_BUDGET if max_chars is None else max(0, int(max_chars))
    limit = min(BLOCK_FIXED + budget, BLOCK_HARD_LIMIT)
    lines = [f"- {m['text']}" for m in picked if m.get("text")]
    while lines:  # 保留 pop：口径万一漂移，真实长度仍是最终裁判
        block = "\n".join([_BLOCK_HEADER, *lines, _BLOCK_FOOTER])
        if len(block) <= limit:
            return block
        lines.pop()
    return ""


def append_block(parts: list[Any], block: str, text_part_cls: Any) -> bool:
    """把注入块追加为临时 content part；本请求已有记忆块则不再重复加。

    去重按 `startswith(_BLOCK_HEADER)` 而不是全等：同轮的抽取任务可能中途写入新记忆，
    内容一变全等就失效，同一请求会出现两个块。
    """
    if not block or text_part_cls is None or parts is None:
        return False
    for part in parts:
        if isinstance(part, text_part_cls) and str(getattr(part, "text", "")).startswith(_BLOCK_TAG):
            return False
    parts.append(text_part_cls(text=block).mark_as_temp())
    return True


# ── 与 store 的编排（异步，但依然不依赖 astrbot） ────────────────────
async def record(store: Any, uid: str, candidates: list[dict[str, Any]], cfg: dict[str, Any]) -> int:
    """写入候选记忆；失败向上抛，由调用方兜。"""
    if not candidates:
        return 0
    async with store.lock(uid):
        items = await store.load(uid)
        merged, changed = merge_items(
            items,
            candidates,
            max_items=cfg_int(cfg, "max_items", 30, 5, 100),
            ttl_days=cfg_int(cfg, "ttl_days", 180, 0, 3650),
        )
        if changed:
            await store.save(uid, merged)
        return len(changed)


async def build_injection(
    store: Any, uid: str, cfg: dict[str, Any], *, query: Any = None
) -> str:
    items = await store.load(uid)
    if not items:
        return ""
    max_items = cfg_int(
        cfg, "inject_max_items", INJECT_MAX_ITEMS_DEFAULT, *INJECT_MAX_ITEMS_RANGE
    )
    max_chars = cfg_int(
        cfg, "inject_max_chars", INJECT_MAX_CHARS_DEFAULT, *INJECT_MAX_CHARS_RANGE
    )
    picked = select_for_injection(
        items, query=query, max_items=max_items, max_chars=max_chars
    )
    # 同一个 max_chars 传给 build_block，保证「选中的一定注入」
    return build_block(picked, max_chars=max_chars)
