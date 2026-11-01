"""物料智能识别服务：把 bidding_records.subject_matter 归并成可检索的"标准物料库"。

问题定义（一物多码）
-------------------
`subject_matter`（标的物）是**采购人自己写的自然语言描述**，同一件物料会有多种写法。
实测 `bidding_records` 的 8746 行里有 **2529 个不同的 subject_matter**，其中大量是
"同一物料 + 不同噪声后缀"：

    打印/复印纸的网上超市采购   160 次  ┐
    打印/复印纸                 88 次  ┘ 同一个物料
    硒鼓                       112 次  ┐
    硒鼓的网上超市采购           58 次  ┘ 同一个物料

后缀 `的网上超市采购` 是**采购渠道**（网上超市）而不是物料属性，必须剥离才能归并。
本模块用两张**可审计的后缀表**（见 NOISE_SUFFIXES / GENERIC_SUFFIXES）加一条
"采购数量"规则（见 _strip_quantity_suffix）做归一化，
实测 2529 → 2272 个标准物料（压缩率 10.2%，见 material_stats）。

金额统计为什么用中位数
----------------------
与 `src/services/analysis.py` 同源的问题：本项目数据**横跨 6 个数量级**
（办公用品几千元 ↔ 基建工程数亿元），均值被少数超大工程完全带偏
（全局均值是中位数的 517 倍）。故本模块所有"典型值"一律用**中位数 / 分位数**，
并且**与均值并排返回**，让前端一眼看出偏态；`median_amount` 绝不可替换为均值。

可疑记录
--------
`analysis.py` 的 `suspicious_sql()` 已经沉淀了"金额 + 品类"组合判据
（托养服务 6.6 亿 = 脏数据；治水工程 4.86 亿 = 真实）。**直接 import 复用**，
不重复实现——两处判据一旦分叉，看板与物料页会给出互相矛盾的结论。

为什么匹配放在 Python 侧（difflib）而不是 pg_trgm
--------------------------------------------------
本机 `pg_trgm` 虽可用（实测 `CREATE EXTENSION` 成功），但**实测效果差，故不采用**：

    A=硒鼓            B=硒鼓的网上超市采购   trgm similarity = 0.182
    A=打印/复印纸      B=打印/复印纸的网上超市采购 trgm = 0.400

`pg_trgm` 是**三元组（trigram）**算法，为英文单词设计。中文没有空格分词，
一个 2 字的物料名只切得出 1 个 trigram，一旦加上后缀，三元组集合几乎不相交，
于是"同一个物料"的相似度反而低于随机两个无关名字。实测同组对照：

    A=硒鼓  B=硒鼓的网上超市采购   ── 字符相似(difflib) 0.364  vs  trgm 0.182
    A=垃圾桶 B=垃圾袋            ── 字符相似(difflib) 0.667  vs  trgm 0.333
                                 （后一对明明**不该**匹配，trgm 却给了更高分）

即 trgm 在本数据上**排序能力是反的**。另有工程上的理由：

    ① `pg_trgm` 默认**未安装**（实测 `pg_available_extensions` 有、`pg_extension` 无），
       需要 `CREATE EXTENSION` 权限；本项目恰好是超级用户所以能装，
       但用户自行部署时未必有该权限，**不能把核心功能压在需要额外授权的扩展上**。
    ② 词汇表只有 2529 个且**几乎不变**，一次性拉进内存后做纯 Python 匹配，
       比每次查询回表 + 走 GIN 索引更快，也不占用数据库连接。

故匹配策略是**分层**的（精确 → 归一化 → 包含 → 字符重叠），
每一层都在 `match_reason` 里如实回报"为什么匹配上"，便于人工核对。
"""

from __future__ import annotations

import re
import threading
import unicodedata
import difflib
from collections import Counter
from typing import Any

from src.datahub.contract import DataHubConfigError as _DataHubConfigError
from src.datahub.pg import PGCursor, dsn as pg_dsn
from src.logging_config import get_logger
from src.services.analysis import suspicious_sql

logger = get_logger(__name__)

TABLE = "bidding_records"
SUBJECT_COL = "subject_matter"


class MaterialError(RuntimeError):
    """物料识别失败（数据库不可用 / 参数不合法 / 物料不存在）。

    对外只暴露这一个异常类型：调用方（API 层）不必了解 psycopg 的异常体系，
    也就不会因为驱动升级换了异常类名而在业务代码里漏捕获。
    """


# ---------------------------------------------------------------------------
# ① 归一化规则
# ---------------------------------------------------------------------------

# 纯噪声后缀：**只表示采购渠道或流程，不携带任何物料属性**，命中即剥离。
#
# 来历：实测 `打印/复印纸的网上超市采购`(160) 与 `打印/复印纸`(88) 是同一物料；
# 全库共 361 个写法含 `网上超市`（1669 行），是本数据集**最大宗的冗余来源**。
#
# ★ 为什么必须保留"的"字（`的网上超市采购` 而非 `网上超市采购`）
# -------------------------------------------------------------
# 中文没有词边界，剥离必须靠**最长后缀精确匹配**。先匹配长的能避免
# "的网上超市"被误留，也避免把"网上超市采购"这个独立写法漏掉。
#
# ★ 为什么是"后缀"而不是"包含即删"
# ---------------------------------
# 数据里有真实反例：`网上超市` 本身就是一个物料写法（8 次，网上超市卡/券类）。
# 若用"包含即删"会把 `网上超市采购` 削成空串。故一律 `endswith` + 长度守卫。
NOISE_SUFFIXES: tuple[str, ...] = (
    "的网上超市采购",
    "网上超市采购",
    "的网上超市",
    "网上超市",
    "采购项目",
    "购置项目",
    "（二次）",
    "(二次)",
    "（重新招标）",
    "(重新招标)",
)

# 通用后缀：**语义上确实是冗余的流程词**，但比 NOISE_SUFFIXES 危险——
# 一旦剥离，"XX采购" 会塌缩成大类 "XX"，可能与已存在的 "XX" 归并。
#
# ★ 实测审查结论：本项目数据上这批剥离**全部正确**，逐条人工核对过的样例
#   （左侧为剥离后，右侧为原始写法，括号内为次数）：
#
#       自粘伤口敷料、聚氨酯泡沫敷料等耗材  ← …耗材采购 (21)
#       检验检测仪器设备                  ← …设备采购 (3)
#       土地复垦EPC                      ← …EPC项目 (2)
#       湖光里休闲民宿(EPC)              ← …（EPC）项目 (1)
#
#   即：`采购` 是动词、`项目` 是流程外壳，**都不描述物料本身**。
#
# ★ 保留字长守卫 MIN_REMAIN_LEN 的原因
# -----------------------------------
# 无守卫时 `网上超市采购` 会塌成 `网上超市`（凑巧正确），但 `采购` 这种
# 纯流程词开头/整体成词的名字（本库共 571 个写法含"采购"）可能被削到 0~1 字，
# 产生无意义的"标准物料"。故剩余长度 < 2 字时**放弃剥离，保留原样**。
GENERIC_SUFFIXES: tuple[str, ...] = (
    "采购",
    "项目",
    "购置",
    "标段",
)

# 剥离后允许的最短长度。见上述说明：2 是"还能辨识是什么物料"的下限。
# 本库最短真实物料是 `空调` / `硒鼓` 这类 2 字词，正好等于该下限。
MIN_REMAIN_LEN = 2

# 迭代剥离顺序：**必须按长度降序**。
#
# 起因（踩过的坑）：`教育类的网上超市采购项目` 同时含 `采购项目`(4 字) 与 `项目`(2 字)。
# 第一版写成"两趟 for + 命中即 break"，剥掉 `采购项目` 就停了，残留的
# `教育类的网上超市` 再没被处理——而那些 `…的网上超市` 写法本可继续归并。
# 按长度降序 + 反复剥离到不动点后，该写法正确归并为 `教育类`。
#
# 注意：NFKC 已把 `（）` 统一成 `()`，故此处只需登记半角形式，
# 登记全角版会得到两个等价项、白白多跑一轮。实测本库无全角 `／`。
_STRIP_ORDER: tuple[str, ...] = tuple(
    sorted(set(NOISE_SUFFIXES) | set(GENERIC_SUFFIXES), key=len, reverse=True)
)

# 空白（含全角空格）一律去除：实测无空格，但外部输入会有。
_WS_RE = re.compile(r"[\s\u3000]+")

# "采购数量"后缀：数字 + 计量单位（详见 _strip_quantity_suffix 的说明）。
# 单位表按实测数据补充——`件/台/套` 是办公设备类最常见的三种。
_QUANTITY_SUFFIX_RE = re.compile(
    r"\d+(?:\.\d+)?(?:台套|件|台|套|个|只|批|箱|包|张|米|吨|组|支|瓶|桶|盒|卷|PCS|pcs|PARTS|pieces)$"
)

# 相邻字符组成的二元组（bigram）。中文里 bigram 比 trigram 可靠：
# 2 字物料名也能切出 1 个 bigram，不会像 trigram 那样退化。
_BIGRAM_CACHE: dict[str, frozenset[str]] = {}


def _bigrams(s: str) -> frozenset[str]:
    """字符二元组集合（缓存：匹配时要对全库 2296 个名字反复算）。"""
    hit = _BIGRAM_CACHE.get(s)
    if hit is None:
        body = s.replace("/", "")
        hit = frozenset(
            body[i : i + 2] for i in range(len(body) - 1)
        ) or frozenset(body)
        _BIGRAM_CACHE[s] = hit
    return hit


def normalize_name(raw: str) -> str:
    """物料名归一化：剥离噪声后缀 + 统一全半角与斜杠 + 去空白。

    处理顺序（**顺序本身是规则的一部分**，颠倒会出错）：

        ① NFKC 全半角统一   —— `Ｕ盘` 与 `U盘` 归并；
                               顺带 `（）`→`()`、`／`→`/`，故斜杠无需单独处理
        ② 去所有空白         —— 中英文物料名里空格只起分隔作用，不携带信息
        ③ 迭代剥离后缀       —— NOISE_SUFFIXES + GENERIC_SUFFIXES，
                               按 _STRIP_ORDER（长度降序）反复剥到不动点
        ④ 大写化             —— `u盘` 与 `U盘` 归并（实测本库两者都存在）

    ★ 为什么不做同义词替换（"其它"→"其他"）
    --------------------------------------
    本库确有 `其它校园家具` / `其他分析仪器` 两种写法。但"其它/其他"是**定冠词**
    而非物料属性，替换后会得到 `其他XX` 这种仍然分叉的名字，收益为负；
    真正该做的是把它们交给 `suggest_standardization` 报给人工确认——
    **归一化只做有确定性的操作，模糊判断交给人**。

    Args:
        raw: 原始物料描述（可为 None / 空串，返回空串而不抛异常，
             因为它的调用方多在"清洗整列数据"的循环里，逐条抛异常无法使用）。

    Returns:
        归一化后的物料名；输入无有效内容时返回 ""。
    """
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", str(raw)).strip()
    s = _WS_RE.sub("", s)
    if not s:
        return ""

    # 迭代剥离到不动点：**循环而非单趟**是踩坑后的修正。
    #
    # 第一版写成"两趟 for + 命中即 break"，`教育类的网上超市采购项目` 只剥到
    # `教育类的网上超市` 就停了（剥掉 `采购项目` 后没回头再试 `的网上超市`），
    # 正确结果是 `教育类`。改为不动点循环：每轮依次尝试"数量后缀 → 噪声后缀"，
    # 只要还能剥就再跑一轮。MIN_REMAIN_LEN 保证循环必然终止。
    changed = True
    while changed:
        changed = False
        stripped = _strip_quantity_suffix(s)
        if stripped != s:
            s, changed = stripped, True
            continue
        for suffix in _STRIP_ORDER:
            if s.endswith(suffix) and len(s) - len(suffix) >= MIN_REMAIN_LEN:
                s = s[: -len(suffix)].strip()
                changed = True
                break

    return s.upper()


def _strip_quantity_suffix(name: str) -> str:
    """剥离末尾的"采购数量"（如 `打印/复印纸250件` → `打印/复印纸`）。

    ★ 为什么要剥离（实测数据支撑）
    ----------------------------
    本库有 30 个写法把**采购数量写进了标的物名**（覆盖 34 条记录）::

        打印/复印纸250件 / 打印/复印纸20件   ← 与 `打印/复印纸`(88) 同一物料
        台式整机183件 / 台式整机1件          ← 与 `台式整机`(131) 同一物料
        空调5件 / 空调8件 / 柜式空调13台      ← 与 `空调`(175) 同一物料

    逐条核对：剥离后**全部**能并入库中已存在的物料（见开发期探针输出），
    没有一条并到错误目标。数量是订单属性，不改变物料身份，属"确定性操作"，
    故放进归一化而非丢给人工。

    ★ 数量模式必须"数字 + 单位"成对（保护型号）
    -----------------------------------------
    只按"末尾是数字"剥会毁掉型号：`3M`、`220P竞赛型…`、`35kV及以上…`、
    `16排及以上螺旋CT` 都以数字或字母结尾。实测含数字但无单位的写法有 35 个，
    全部**保持原样**（正则要求末尾必须是数量单位，型号词不匹配）。

    ★ 已知局限（有意不修）
    --------------------
    `自粘伤口敷料、聚氨酯泡沫敷料等耗材采购项目03包` 这类**包号**（`03包`）
    虽然匹配 `数量+包`，但剥完会剩下半截 `…采购项目第` 这样的残句。
    本函数不做特殊处理：包号确实该并入主物料，但"第N包/0N包"的形态多样，
    硬编码规则容易伤及真实物料；交给 `suggest_standardization` 报人工确认，
    与 `analysis.py`"系统负责发现、人负责决策"的思路一致。
    """
    return _QUANTITY_SUFFIX_RE.sub("", name)


# ---------------------------------------------------------------------------
# ② 数据库连接（与 analysis.py 同款：连库 → 用 → 必关）
# ---------------------------------------------------------------------------


def _dsn() -> dict[str, Any]:
    """兼容包装：DSN 组装已收口到 `src.datahub.pg`。"""
    try:
        return pg_dsn()
    except _DataHubConfigError as exc:
        raise MaterialError(str(exc)) from exc


def _Cursor():
    """兼容包装：连接/超时/异常映射统一由 `src.datahub.pg.PGCursor` 负责。"""
    return PGCursor(MaterialError)


def _fetchall(cur, sql: str, params: tuple = ()) -> list[tuple]:
    """执行查询并取回全部行，**把驱动异常收成 MaterialError**。

    ★ 为什么 `_Cursor` 之外还要这一层
    --------------------------------
    `_Cursor.__enter__` 只负责"连不上"这一种失败；连上之后语句本身出错
    （表被删、权限不足、SQL 写错）抛的仍是 `psycopg.errors.*`。
    若不在出口处收敛，调用方（API 层）就必须 import psycopg 才能写 except，
    驱动一升级换异常类名就会漏捕获。故**所有** SQL 都经此函数执行。
    """

    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception as exc:
        raise MaterialError(f"查询失败：{type(exc).__name__}") from exc


def _fetchone(cur, sql: str, params: tuple = ()) -> tuple | None:
    """同 `_fetchall`，只取第一行。"""
    try:
        cur.execute(sql, params)
        return cur.fetchone()
    except Exception as exc:
        raise MaterialError(f"查询失败：{type(exc).__name__}") from exc


def _amount_filter(alias: str = "") -> str:
    """金额有效性条件：非空、为正，且**排除可疑记录**。

    复用 `analysis.suspicious_sql()` —— 见模块 docstring"可疑记录"一节。
    """
    col = f"{alias}.amount" if alias else "amount"
    subj = f"{alias}.{SUBJECT_COL}" if alias else SUBJECT_COL
    return (
        f"{col} IS NOT NULL AND {col} > 0 "
        f"AND NOT {suspicious_sql(col, subj)}"
    )


# ---------------------------------------------------------------------------
# ③ 标准物料库索引（进程内缓存）
# ---------------------------------------------------------------------------

# 缓存有效期：进程生命周期。
#
# 为什么敢长期缓存：subject_matter 来自离线入库的标讯数据，**不是实时写入**，
# 变化频率以天计；而每次冷启动要拉 2529 行 × 全部分组，约 200ms。
# 若将来改为实时入库，调用 `clear_cache()` 即可失效。
_INDEX_CACHE: dict[str, Any] | None = None
# FastAPI 默认多线程跑同步端点，两个请求可能同时触发冷启动构建，
# 无锁会重复拉库（虽不致命，但白白多两次全表聚合）。
_INDEX_LOCK = threading.Lock()

# 相似度低于此值的候选直接丢弃，避免"什么都匹配得上"。
#
# ★ 这个门槛**同时管住"包含匹配"与"字符重叠"两层**（踩过的坑）
# ---------------------------------------------------------
# 起初只把它用在"字符重叠"层，结果 `支架` ⊂ `麦克风/话筒支架` 以 0.12 分
# 挤进了候选列表——因为包含层的公式是无界的（见 _containment_score），
# "多出部分远长于短名"时会算出很低的分，却仍被当成有效包含返回。
# 现在两层共用同一门槛，`min(score)` 即"这是不是同一物料"的唯一分界线。
#
# 取值依据（实测）：中文短串的相似度天然偏高，必须把门槛抬高。
# 用 `0.4*序列比 + 0.6*bigram` 实测得到：
#
#     硒鼓 ~ 硒鼓的网上超市采购   0.364   ← 同一物料，但②③层已先命中，无需④兜底
#     空调    ~ 空调机          0.640   ← 会被丢弃，可接受（"机"字差异另有专名）
#     配电工程 ~ 供配电工程       0.806   ← 不同类别，包含层已压到 0.79 丢弃
#     台式整机 ~ 台式计算机       0.629   ← 不同物料，**必须丢弃**
#     垃圾桶  ~ 垃圾袋           0.667   ← 不同物料，**必须丢弃**
#
# 故门槛定在 0.80：只有"错别字 / 字序调换 / 近义改写 / 定语极少的包含"才通过。
# 宁可少给候选，也不要给错误候选——用户看到 `垃圾桶 → 垃圾袋` 会直接不信任本功能。
MIN_MATCH_SCORE = 0.80

# "包含匹配"允许的最大额外长度（见 _containment_score）。
# `麦克风/话筒` ⊂ `麦克风/话筒支架` 额外 2 字、`医学检验试剂、耗材` ⊂ `…耗材等` 额外 1 字；
# 超过 6 字的多半是"泛化短名 + 一长串描述"，不足以断定同一物料。
MAX_QUALIFIER_LEN = 6

# 走"包含匹配"层要求短名至少这么长。
#
# ★ 为什么是 2（第一版设 3，回退过）
# --------------------------------
# 曾担心 `配电工程` ⊂ `供配电工程`、`视频设备` ⊂ `音视频设备` 这类假配对
# （多出的字**加在短名前**，语义已变），于是把门槛提到 3 字以阻断 2 字短名。
# 但实测这**误伤**了真包含：`空调` ⊂ `空调配件` 正是采购人最常查的写法之一，
# 被挡掉会让功能明显不可用。
#
# 正确做法是**保留包含关系但降低置信度**（见 _containment_is_insertion）：
# 2 字短名照样能匹配上，只是不会被打上"高置信"标签。故门槛回到 2。
MIN_CONTAIN_LEN = 2

# 打分档位表（**分数区间刻意不重叠**，故排序即置信度排序，前端可直接按序展示）。
#
# 各档取值的依据：
#   精确匹配 1.00   —— 无歧义
#   归一化匹配 0.95 —— 剥后缀后完全一致，是本模块的核心价值，仅次于精确
#   包含匹配 ≤0.94  —— 见 _containment_score 的公式说明
#   字符重叠 ≤0.90  —— 由 MIN_MATCH_SCORE 把住下限
SCORE_EXACT = 1.0
SCORE_NORMALIZED = 0.95

# 包含匹配打分：
#     0.90 − 0.08*(多出字数/短名字数) − 0.09*前置字数 − 0.02*后置字数
#
# 前两个版本的失败原因与实测数据见 _containment_score 的文档字符串。要点：
#   ① 用**比例**而非绝对字数 —— 给 30 字的长名加 2 字（`…(EPC)`→`…(EPC)监理`），
#      新增信息远少于给 4 字短名加 1 字（`配电工程`→`供配电工程`）
#   ② **前置罚得极重（0.09/字），后置很轻（0.02/字）** —— 实测这组权重才把
#      `配电工程`⊂`供配电工程`(0.76) 压到真候选 `医学检验试剂、耗材`⊂`…耗材等`(0.87) 之下；
#      前置 0.04 时两者是 0.84 vs 0.87，几乎并排，人工队列无法排序
#   ③ 基准 0.90 低于 SCORE_NORMALIZED(0.95)，保证"只是包含"排不进"归一化匹配"之前
CONTAIN_BASE = 0.90
CONTAIN_EXCESS_WEIGHT = 0.08
CONTAIN_HEAD_PENALTY = 0.09
CONTAIN_TAIL_PENALTY = 0.02
SCORE_OVERLAP_FACTOR = 0.9

# 包含匹配的最高分：必须严格低于"归一化后匹配"(0.95)，
# 否则"只是包含"会和"剥掉后缀后完全一致"混为一谈。
CONTAIN_MAX_SCORE = 0.94

DEFAULT_MATCH_LIMIT = 10
MAX_MATCH_LIMIT = 50

DEFAULT_SUGGEST_LIMIT = 50
MAX_SUGGEST_LIMIT = 200

# 归并建议的相似度下限：高于此值但**尚未**被归一化规则合并的，才值得人工核。
#
# 取值依据（实测 42 对候选，覆盖全部形态）：
#   0.896  字符重叠   …工程项目第一包 ~ …工程项目第二包      ← 招标包号，应按包拆而非并
#   0.882  包含匹配   搬迁安置工程…总承包(EPC) ~ …(EPC)监理  ← 真候选
#   0.870  包含匹配   医学检验试剂、耗材 ~ 医学检验试剂、耗材等 ← 真候选
#   0.850  包含匹配   球型摄像机 ~ 半球型摄像机             ← 假候选（中间插字）
#   0.825  包含匹配   羽毛球 ~ 羽毛球拍                     ← 假候选（球 vs 球拍）
#   0.814  包含匹配   笔记本电脑 ~ 采购笔记本电脑            ← 真候选
#
# 结论：0.80 保留 42 对，**人工一次就能看完**，符合"宁可少报"的定位。
# 设到 0.85 会漏掉真候选；设到 0.70 会涌入大批 `XX配件`/`XX服务` 噪声。
SUGGEST_MIN_SCORE = 0.80


def _clamp(value: Any, low: int, high: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = low
    return max(low, min(high, n))


def clear_cache() -> None:
    """清空标准物料库缓存（数据重新入库后调用）。"""
    global _INDEX_CACHE
    with _INDEX_LOCK:
        _INDEX_CACHE = None


def _build_index() -> dict[str, Any]:
    """构建标准物料库索引（一次性拉全表 + 归一化 + 分组）。

    ★ 为什么把归一化放在 Python 而不是写 SQL
    --------------------------------------
    ① 归一化规则（后缀表）写在 Python 常量里可读、可测、可 diff；
       写成 SQL 的嵌套 `regexp_replace` 无法审查，改一条规则要重写整段 SQL。
    ② 归一化后还要做**分组聚合**，SQL 里得再套一层子查询 + GROUP BY，
       复杂度和调试成本都高于在 2529 行的规模上直接算。
    ③ 未来把归一化换成模型/词表时，只替换本函数即可，SQL 不用动。

    返回结构::

        {
          "variants":  {标准名: [(原始写法, 次数), ...]},   # 按次数降序
          "counts":    {标准名: 总次数},
          "raw_total": 原始写法总数,
          "raw_rows":  参与归并的记录数,
          "lookup":    {归一化写法: 标准名},                # 反查用
        }
    """
    with _Cursor() as cur:
        rows = _fetchall(
            cur,
            f"""
            SELECT {SUBJECT_COL}, count(*) AS cnt
            FROM {TABLE}
            WHERE {SUBJECT_COL} IS NOT NULL AND {SUBJECT_COL} <> ''
            GROUP BY {SUBJECT_COL}
            ORDER BY cnt DESC
            """,
        )

    variants: dict[str, list[tuple[str, int]]] = {}
    lookup: dict[str, str] = {}
    for raw, cnt in rows:
        key = normalize_name(raw) or str(raw).strip().upper()
        variants.setdefault(key, []).append((raw, int(cnt)))
        # 反查表同时登记"原始写法"和"归一化写法"，两者都可能被用户输入：
        #   输入 `硒鼓的网上超市采购` → 归一到 `硒鼓`       → 命中标准名 `硒鼓`
        #   输入 `硒鼓`               → 归一到 `硒鼓`       → 命中标准名 `硒鼓`
        lookup.setdefault(key, key)
        lookup.setdefault(str(raw).strip().upper(), key)

    for group in variants.values():
        group.sort(key=lambda item: -item[1])

    index = {
        "variants": variants,
        "counts": {k: sum(c for _, c in v) for k, v in variants.items()},
        "raw_total": len(rows),
        "raw_rows": sum(int(r[1]) for r in rows),
        "lookup": lookup,
    }
    logger.info(
        "标准物料库已构建：%d 个原始写法 → %d 个标准物料",
        index["raw_total"],
        len(variants),
    )
    return index


def _index() -> dict[str, Any]:
    """取索引（首次调用时构建，之后走进程内缓存）。"""
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        with _INDEX_LOCK:
            if _INDEX_CACHE is None:  # 双重检查：只有一个线程真正构建
                _INDEX_CACHE = _build_index()
    return _INDEX_CACHE


def _resolve(standard_name: str) -> str | None:
    """把用户给的任意写法解析成标准物料名（对不上返回 None）。"""
    if not standard_name:
        return None
    index = _index()
    key = normalize_name(standard_name) or str(standard_name).strip().upper()
    # 先查反查表（覆盖"原始写法"与"归一化写法"两种输入），
    # 再退回 variants（归一化结果可能不等于任何原始写法，但仍是合法标准名）
    hit = index["lookup"].get(key)
    if hit is not None:
        return hit
    return key if key in index["variants"] else None


def _variants_of(standard_name: str) -> list[str]:
    """标准物料名 → 该组下所有原始写法（用于 SQL 里 IN 查询）。"""
    index = _index()
    group = index["variants"].get(standard_name)
    if not group:
        return []
    # 除原始写法外，也带上归一化写法本身——归一化结果不一定恰好等于某个原始写法
    names = [raw for raw, _ in group]
    if standard_name not in names:
        names.append(standard_name)
    return names


# ---------------------------------------------------------------------------
# ④ 金额工具
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], q: float) -> float | None:
    """线性插值分位数（与 PostgreSQL `percentile_cont` 同算法）。

    ★ 为什么不直接 `statistics.quantiles`
    -----------------------------------
    `statistics.quantiles` 用"排位取点"法，n 较小时与 PostgreSQL 的
    `percentile_cont` 结果不一致。本模块既然在 SQL 里也用 `percentile_cont`，
    就必须保证两处口径统一，否则同一物料在两个接口里给出不同中位数。

    调用方**必须传入已排序**的列表（本函数为热路径，不重复排序）。
    """
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo]) * (1 - frac) + float(sorted_values[hi]) * frac


def _median(values: list[float]) -> float | None:
    """中位数。**所有金额典型值都必须走这里**，理由见模块 docstring。"""
    return _percentile(sorted(values) if values else [], 0.5)


# ---------------------------------------------------------------------------
# ⑤ 匹配打分
# ---------------------------------------------------------------------------


def _similarity(a: str, b: str) -> float:
    """字符相似度 0~1：difflib 序列比 与 字符 bigram Jaccard 的加权融合。

    ★ 为什么融合两个指标而不是只用 difflib
    ------------------------------------
    实测两者各有盲区（A=查询，B=库内标准物料）::

        A=空调    B=空调机        序列比 0.800  bigram 0.500   ← 序列比偏乐观
        A=台式整机 B=台式计算机    序列比 0.667  bigram 0.167   ← 实为不同物料
        A=垃圾桶  B=垃圾袋        序列比 0.667  bigram 0.333   ← 实为不同物料

    `difflib.SequenceMatcher.ratio()` 对**短中文串过于宽松**：2~3 字的名字
    共享 1 个字就能拿到 0.5+。bigram Jaccard 更严格，但对"多一个后缀"
    这类整体包含的情况又会低估（`打印/复印纸` vs `打印/复印纸的网上超市采购`
    只有 0.364）。

    故取 `0.4*序列比 + 0.6*bigram`：以 bigram 为主（严格、抗短串噪声），
    序列比作补充（对错别字、字序调换更宽容）。权重是实测调出来的，
    不是理论最优——本数据上它让"同物料"稳定高于 0.6、"不同物料"低于 0.45。
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    seq = difflib.SequenceMatcher(None, a, b).ratio()
    ba, bb = _bigrams(a), _bigrams(b)
    union = ba | bb
    jac = len(ba & bb) / len(union) if union else 0.0
    return 0.4 * seq + 0.6 * jac


def _containment_parts(query: str, standard: str) -> tuple[str, str, str, str, str] | None:
    """拆解包含关系，返回 (短名, 长名, 多出部分, 前置部分, 后置部分)。

    不构成包含（或超出 MAX_QUALIFIER_LEN）时返回 None。

    ★ 为什么必须区分"前置"与"后置"
    -----------------------------
    多出的字加在**前面**还是**后面**，语义后果完全不同::

        加在后面：`医学检验试剂、耗材` → `…耗材等`     纯后缀/量词，几乎不改变语义
        加在前面：`配电工程` → `供配电工程`           前缀是**类别限定**（供配电≠配电）
                  `热水壶`  → `电热水壶`             前缀是**动力方式**（电热≠普通）

    实测把两者混为一谈时，真包含与假包含的得分只差 0.03，人工队列无法排序。
    拆开后可以对"前置"罚得更重，让 `配电工程`/`供配电工程` 这类沉到队列后面。
    """
    if not query or not standard or query == standard:
        return None
    short, long_ = (query, standard) if len(query) <= len(standard) else (standard, query)
    if short not in long_:
        return None
    idx = long_.find(short)
    head = long_[:idx]
    tail = long_[idx + len(short) :]
    extra = head + tail
    if not extra or len(extra) > MAX_QUALIFIER_LEN:
        return None
    return short, long_, extra, head, tail


def _containment_score(query: str, standard: str) -> float:
    """包含关系得分：短名是长名的一部分，多出的部分越少，越可能是同一物料。

    （注：`硒鼓` ⊂ `硒鼓的网上超市采购` 这类通常轮不到本层——归一化已把后缀剥掉，
    直接命中"归一化后匹配"。包含层真正服务的是归一化处理不了的定语，
    如 `空调` ⊂ `空调配件`、`医学检验试剂、耗材` ⊂ `医学检验试剂、耗材等`。）

    ★ 打分公式的由来（三版都踩过坑，记录在此以免回退）
    -----------------------------------------------
    第一版按**多出字数**统一扣分 → `配电工程`⊂`供配电工程`（多 1 字）与
    `医学检验试剂、耗材`⊂`…耗材等`（多 1 字）得分相同，**分不开**。

    第二版按**长名覆盖率**上浮 → 两者只差 0.03，且长名反而吃亏：
    `搬迁安置…(EPC)`⊂`…(EPC)监理` 的覆盖率高达 94%，得分却只有 0.814，
    不如短名的 0.850。原因是长名里"未变化的部分"太多，把新增信息稀释了。

    第三版（当前）改用**多出部分占短名的比例** `len(extra)/len(short)`：

        配电工程 / 供配电工程                  多 1 字 / 短名 4 字  = 0.25
        医学检验试剂、耗材 / …耗材等            多 1 字 / 短名 9 字  = 0.11
        搬迁安置…(EPC) / …(EPC)监理            多 2 字 / 短名 30 字 = 0.07

    这个量直接回答"**相对这个物料名，新增了多少信息**"，与名字长短无关：
    给 30 字的长名加 2 字（"监理"）显然比给 4 字短名加 1 字（"供"）更不像换了个物料。
    实测排序立刻正确（长名前缀包含 0.851 > 短名插字包含 0.757）。

    再加上**前置比后置罚得重**（前置多是类别限定，见 _containment_parts 的说明），
    `配电工程`⊂`供配电工程` 进一步压到 0.757，与真候选拉开 0.09 以上，
    同时仍在 SUGGEST_MIN_SCORE(0.80) 之下——即**它连人工队列都不该进**。

    结论：**分数只提供连续信号，真正的分档由 `_containment_is_insertion`
    的结构判据完成**——不要把判断力全压在一个阈值上。
    """
    parts = _containment_parts(query, standard)
    if parts is None:
        return 0.0
    short, _long, extra, head, tail = parts
    excess_ratio = len(extra) / len(short)
    return (
        CONTAIN_BASE
        - CONTAIN_EXCESS_WEIGHT * excess_ratio
        - CONTAIN_HEAD_PENALTY * len(head)
        - CONTAIN_TAIL_PENALTY * len(tail)
    )


def _containment_is_insertion(query: str, standard: str) -> bool:
    """多出来的部分是"确定性不足"的形态，置信度需要降档（见 _suggest_pair）。

    两类被判为"需重点核实"：

        ① 短名 ≥3 字但多出的字**两边都有**（真正插在中间）
        ② 短名仅 2 字（`空调` ⊂ `空调配件`）—— 2 字名过于宽泛，
           一个定语就足以把它变成另一件东西

    ★ 已知局限：`热水壶` ⊂ `电热水壶`、`配电工程` ⊂ `供配电工程`
    ---------------------------------------------------------
    这类"单字前置"是否改变物料身份，**字符串层面无法判定**
    （`热水壶`→`电热水壶` 行业里常视为同一物料，`配电工程`→`供配电工程` 则是两类工程）。
    本函数不做不可靠的猜测，改为返回 False 并在 `_suggest_pair` 里
    用 `head_modifier` 单独标注，让它排在队列后段由人工判断。
    """
    parts = _containment_parts(query, standard)
    if parts is None:
        return False
    short, long_, _extra, head, tail = parts
    if len(short) <= MIN_CONTAIN_LEN:
        return True
    return bool(head) and bool(tail)


def _match_one(query: str, standard: str, norm_query: str, norm_standard: str) -> tuple[float, str]:
    """单条候选打分，返回 (得分, 匹配原因)。得分 0 表示不匹配。"""
    # ① 精确匹配：用户输入与标准名完全一致（含大小写）
    if query == standard:
        return SCORE_EXACT, "精确匹配"

    # ② 归一化后匹配：剥掉后缀/统一大小写后一致 —— 本模块的核心价值所在
    if norm_query and norm_query == norm_standard:
        return SCORE_NORMALIZED, "归一化后匹配"

    # ③ 包含匹配：短名是长名的一部分（见 _containment_score）
    #
    # ★ 必须和④用同一个门槛（踩过的坑）
    # ---------------------------------
    # 包含层的公式是**无界**的：`0.90 - 0.08*(多出/短名) - 0.09*前置 - 0.02*后置`
    # 当"多出的部分"远长于短名时（`支架` ⊂ `麦克风/话筒支架`，多出 5 字全是前置），
    # 得分会跌到 0.12 —— 这本是**正确**的信号（`支架` 显然不等于 `麦克风/话筒支架`），
    # 但第一版漏了门槛，导致 0.12 分的弱包含照样被当作候选返回。
    # 故此处与④统一用 MIN_MATCH_SCORE 卡一道。
    contain = _containment_score(norm_query, norm_standard)
    if contain >= MIN_MATCH_SCORE:
        return round(min(CONTAIN_MAX_SCORE, contain), 4), "包含匹配"

    # ④ 字符重叠：错别字、简称、字序调换的最后一道网。
    # 阈值取得高（MIN_MATCH_SCORE）：这一层没有结构约束，
    # 放宽会立刻放进 `垃圾桶`~`垃圾袋` 这类"像但不是"的候选。
    sim = _similarity(norm_query, norm_standard)
    if sim >= MIN_MATCH_SCORE:
        return round(SCORE_OVERLAP_FACTOR * sim, 4), "字符重叠"
    return 0.0, ""


# ---------------------------------------------------------------------------
# ⑥ 对外接口 1/5：语义/模糊匹配
# ---------------------------------------------------------------------------


def match_materials(query: str, limit: int = DEFAULT_MATCH_LIMIT) -> dict[str, Any]:
    """输入任意物料描述，返回候选标准物料列表（按得分降序）。

    匹配分四层，每层命中即返回对应 `match_reason`（见 `_match_one`）::

        精确匹配    1.00   输入与标准名完全一致
        归一化后匹配 0.95   剥掉"的网上超市采购"等后缀后一致
        包含匹配    0.85+  一方是另一方子串，按长度占比上浮
        字符重叠    0~0.90 序列比 + bigram 融合分，兜错别字/简称

    每层得分**区间不重叠**，故排序即置信度排序，前端可直接按序展示。

    ★ 为什么 `median_amount` 用中位数而不是均值
    ----------------------------------------
    一个标准物料下可能同时有 5000 元的办公耗材和 2 亿元的工程
    （本数据集量级横跨 6 个数量级），均值会被单条超大额记录完全带偏。
    故主字段是 `median_amount`，**同时**返回 `mean_amount` 供对照——
    两者差距过大本身就是"该物料下混入了不同量级的东西"的信号。

    Args:
        query: 用户输入的物料名（可以是错的 / 简写的 / 带后缀的）。
        limit: 返回条数，默认 10，上限 50。

    Returns:
        ``{"query", "normalized_query", "matched", "total_candidates", "items"}``，
        `items` 元素含 `standard_name` / `matched_variants` / `total_count` /
        `median_amount` / `mean_amount` / `score` / `match_reason`。
    """
    text = (query or "").strip()
    if not text:
        raise MaterialError("请输入物料描述")

    n = _clamp(limit, 1, MAX_MATCH_LIMIT)
    norm_query = normalize_name(text)
    index = _index()
    variants = index["variants"]

    scored: list[tuple[float, str, str]] = []
    for standard in variants:
        score, reason = _match_one(text, standard, norm_query, standard)
        if score > 0:
            scored.append((score, reason, standard))

    scored.sort(key=lambda item: (-item[0], -index["counts"][item[2]], item[2]))
    top = scored[:n]

    # 只为命中的候选取金额，避免为全库 2296 个物料各查一次
    amounts = _amounts_for([std for _, _, std in top]) if top else {}

    items = []
    for score, reason, standard in top:
        group = variants[standard]
        vals = amounts.get(standard, [])
        items.append(
            {
                "standard_name": standard,
                "matched_variants": [
                    {"name": raw, "count": cnt} for raw, cnt in group
                ],
                "total_count": index["counts"][standard],
                "median_amount": _median(vals),
                "mean_amount": (sum(vals) / len(vals)) if vals else None,
                "amount_samples": len(vals),
                "score": round(score, 4),
                "match_reason": reason,
            }
        )

    return {
        "query": text,
        "normalized_query": norm_query,
        "matched": bool(items),
        "total_candidates": len(scored),
        "returned": len(items),
        "items": items,
        "note": (
            "金额统计使用中位数并已排除可疑记录；"
            "mean_amount 仅供参考，与中位数差距大说明该物料下混有不同量级的项目"
        ),
    }


def _amounts_for(standard_names: list[str]) -> dict[str, list[float]]:
    """批量取若干标准物料下的全部有效金额（用于算中位数/分位数）。

    一次查完而不是逐个物料查：匹配接口一次要返回 10 个候选，
    逐个查就是 10 次往返。`= ANY(%s)` 让 PostgreSQL 用一次索引扫描完成。
    """
    if not standard_names:
        return {}
    index = _index()
    wanted: dict[str, list[str]] = {}
    all_names: list[str] = []
    for std in standard_names:
        names = _variants_of(std)
        wanted[std] = names
        all_names.extend(names)

    result: dict[str, list[float]] = {std: [] for std in standard_names}
    with _Cursor() as cur:
        amount_rows = _fetchall(
            cur,
            f"""
            SELECT {SUBJECT_COL}, amount
            FROM {TABLE}
            WHERE {SUBJECT_COL} = ANY(%s) AND {_amount_filter()}
            """,
            (all_names,),
        )
    # 原始写法 → 标准名的反查（一个原始写法只属于一个标准物料）
    owner = {raw: std for std, raws in wanted.items() for raw in raws}
    for raw, amount in amount_rows:
        std = owner.get(raw)
        if std is not None and amount is not None:
            result[std].append(float(amount))
    return result


# ---------------------------------------------------------------------------
# ⑦ 对外接口 2/5：单个标准物料详情
# ---------------------------------------------------------------------------


def material_detail(standard_name: str) -> dict[str, Any]:
    """单个标准物料的画像：分位数 + 主要供应商 / 采购人 / 地区。

    交易次数与分位数都**跨该标准物料的所有原始写法**统计——
    这正是归并的意义：只看 "硒鼓" 会漏掉 "硒鼓的网上超市采购" 的 58 条记录。

    ★ 为什么供应商/采购人用"次数"而不是"金额"排序
    -------------------------------------------
    金额排序会被单条超大额记录主导（一条 2 亿的工程直接压过 100 条耗材），
    而本接口回答的是"**谁在经常买/卖这个物料**"，是频次问题不是金额问题。
    金额中位数作为附加字段给出，用于识别"这家供应商专做大单"这类模式。
    """
    key = (standard_name or "").strip()
    if not key:
        raise MaterialError("请提供标准物料名")

    resolved = _resolve(key)
    if resolved is None:
        # 未精确命中标准名时，退回模糊匹配，给出最接近的候选而不是直接失败
        fallback = match_materials(key, limit=1)
        if not fallback["items"]:
            return {
                "standard_name": key,
                "matched": False,
                "message": f"物料库中没有与「{key}」匹配的标准物料",
            }
        resolved = fallback["items"][0]["standard_name"]

    index = _index()
    names = _variants_of(resolved)
    with _Cursor() as cur:
        row = _fetchone(
            cur,
            f"""
            SELECT
                count(*),
                percentile_cont(0.1) WITHIN GROUP (ORDER BY amount),
                percentile_cont(0.25) WITHIN GROUP (ORDER BY amount),
                percentile_cont(0.5)  WITHIN GROUP (ORDER BY amount),
                percentile_cont(0.75) WITHIN GROUP (ORDER BY amount),
                percentile_cont(0.9)  WITHIN GROUP (ORDER BY amount),
                min(amount), max(amount), avg(amount)
            FROM {TABLE}
            WHERE {SUBJECT_COL} = ANY(%s) AND {_amount_filter()}
            """,
            (names,),
        )
        amount_rows = int(row[0] or 0)

        suppliers = _fetchall(
            cur,
            f"""
            SELECT supplier, count(*) c,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY amount)
            FROM {TABLE}
            WHERE {SUBJECT_COL} = ANY(%s) AND {_amount_filter()}
              AND supplier IS NOT NULL AND supplier <> ''
            GROUP BY supplier ORDER BY c DESC, supplier LIMIT 8
            """,
            (names,),
        )

        purchasers = _fetchall(
            cur,
            f"""
            SELECT purchaser, count(*) c
            FROM {TABLE}
            WHERE {SUBJECT_COL} = ANY(%s) AND {_amount_filter()}
              AND purchaser IS NOT NULL AND purchaser <> ''
            GROUP BY purchaser ORDER BY c DESC, purchaser LIMIT 8
            """,
            (names,),
        )

        regions = _fetchall(
            cur,
            f"""
            SELECT COALESCE(NULLIF(location_city, ''), NULLIF(province, '')) AS region,
                   count(*) c
            FROM {TABLE}
            WHERE {SUBJECT_COL} = ANY(%s) AND {_amount_filter()}
            GROUP BY region ORDER BY c DESC NULLS LAST LIMIT 8
            """,
            (names,),
        )

        # 总记录数（不排除可疑金额），用于说明"有多少条因金额可疑被排除"
        total_row = _fetchone(
            cur,
            f"SELECT count(*) FROM {TABLE} WHERE {SUBJECT_COL} = ANY(%s)",
            (names,),
        )
        total_rows = int(total_row[0] or 0)

    def f(v):
        return float(v) if v is not None else None

    p10, p25, p50, p75, p90 = (f(row[i]) for i in range(1, 6))
    mean = f(row[8])

    caveats: list[str] = []
    if amount_rows == 0:
        caveats.append("该物料没有金额有效的记录，无法给出价格参考")
    if mean is not None and p50 and mean > p50 * 3:
        caveats.append(
            f"金额极度偏态：均值 ¥{mean:,.0f} 是中位数 ¥{p50:,.0f} 的 "
            f"{mean / p50:.0f} 倍，**请以中位数判断典型水平**"
        )
    if amount_rows < 5:
        caveats.append(f"仅 {amount_rows} 条有效记录，样本偏少，结论仅供参考")
    excluded = total_rows - amount_rows
    if excluded > 0:
        caveats.append(
            f"另有 {excluded} 条记录被排除（金额为空/非正或金额可疑，"
            f"可疑判据见 analysis.suspicious_sql）"
        )

    return {
        "standard_name": resolved,
        "requested": key,
        "matched": True,
        "total_count": index["counts"].get(resolved, 0),
        "amount_count": amount_rows,
        "matched_variants": [
            {"name": raw, "count": cnt} for raw, cnt in index["variants"][resolved]
        ],
        "percentiles": {
            "p10": p10, "p25": p25, "median": p50, "p75": p75, "p90": p90,
        },
        "min_amount": f(row[6]),
        "max_amount": f(row[7]),
        "mean_amount": mean,
        "suggested_range": {"low": p25, "high": p75},
        "top_suppliers": [
            {"name": r[0], "count": r[1], "median_amount": f(r[2])} for r in suppliers
        ],
        "top_purchasers": [{"name": r[0], "count": r[1]} for r in purchasers],
        "top_regions": [{"name": r[0] or "未知", "count": r[1]} for r in regions],
        "data_caveats": caveats,
    }


# ---------------------------------------------------------------------------
# ⑧ 对外接口 3/5：物料库总览
# ---------------------------------------------------------------------------


def material_stats(top_n: int = 20) -> dict[str, Any]:
    """物料库总览：归并前后规模、压缩率、交易次数 TOP N。

    ★ "压缩率"的口径必须写清楚
    -------------------------
    压缩率 = 1 - 标准物料数 / 原始写法数，衡量的是**同一个物料有多少种写法**。
    它不是"去重率"，也不代表"节省了多少存储"——本模块**不修改原表数据**，
    原始写法全部保留在 `matched_variants` 里可追溯。
    之所以要坚持这个口径：一旦有人把 9.2% 读成"9.2% 的数据是脏的"，
    就会得出与实际相反的结论（真实情况是同一物料平均有 1.1 种写法）。
    """
    n = _clamp(top_n, 1, 100)
    index = _index()
    counts = index["counts"]
    standard_total = len(counts)
    raw_total = index["raw_total"]

    with _Cursor() as cur:
        table_row = _fetchone(cur, f"SELECT count(*) FROM {TABLE}")
        table_rows = int(table_row[0] or 0)

    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    top_amounts = _amounts_for([std for std, _ in top])

    merged = sum(1 for g in index["variants"].values() if len(g) > 1)
    merged_rows = sum(
        sum(c for _, c in g) for g in index["variants"].values() if len(g) > 1
    )

    return {
        "standard_total": standard_total,
        "raw_total": raw_total,
        "compression_ratio": round(1 - standard_total / raw_total, 4) if raw_total else 0.0,
        "merged_groups": merged,
        "merged_rows": merged_rows,
        "table_rows": table_rows,
        "empty_subject_rows": table_rows - index["raw_rows"],
        "top_materials": [
            {
                "standard_name": std,
                "total_count": cnt,
                "variant_count": len(index["variants"][std]),
                "median_amount": _median(top_amounts.get(std, [])),
            }
            for std, cnt in top
        ],
        "notes": [
            f"原始写法 {raw_total} 个 → 标准物料 {standard_total} 个，"
            f"压缩率 {round((1 - standard_total / raw_total) * 100, 1) if raw_total else 0}%"
            "（口径：同一物料的写法数，非数据脏污率）",
            f"其中 {merged} 个标准物料由 ≥2 种原始写法归并而成，覆盖 {merged_rows} 条记录",
            "TOP 排序按交易次数；金额为中位数，已排除可疑记录",
        ],
    }


# ---------------------------------------------------------------------------
# ⑨ 对外接口 4/5：归并建议（人工确认）
# ---------------------------------------------------------------------------

# 建议分档：给人工一个"先看哪条"的优先级。
#   高：≥0.84 —— 定语加在短名**之后**（`…耗材` → `…耗材等`），基本不改变物料身份
#   中：≥SUGGEST_MEDIUM —— 可能是同一物料，但存在"定语改变类别"的可能
#   低：≥SUGGEST_MIN_SCORE —— 需逐条判断
#
# ★ 为什么敢把 0.90 降到 0.84
# --------------------------
# 起初担心"高置信"被假配对污染，故门槛设 0.90。实测**一对都进不去**
# （最高 0.896），"高"档形同虚设；同时 `医学检验试剂、耗材`⊂`…耗材等`（0.850）
# 这种明显该优先处理的被判成"中"，等于没有优先级。
#
# 真正的把关者不是分数而是 `head_modifier` / `ambiguous` 结构标记：
# `配电工程`⊂`供配电工程`（0.820）虽在分数上接近，但带 head_modifier 会被压到"中"。
# 故分数门槛可以放宽，**结构标记负责兜底**。
SUGGEST_HIGH = 0.84
SUGGEST_MEDIUM = 0.80


def _suggest_pair(a: str, b: str, ca: int, cb: int) -> dict[str, Any] | None:
    """判断两个标准物料是否构成"疑似应归并"的一对。

    只保留两类信号，**其余一律不报**（宁可少报，也不要用噪声淹没人工）::

        ① 包含关系：`复印机` ⊂ `复印机配件`
        ② 高字符相似：`高标准农田建设` ~ `高标准农田提升建设`

    ★ 为什么包含档也用 SUGGEST_MIN_SCORE 卡一遍
    -----------------------------------------
    `_containment_score` 对"疑似插字"（`配电工程` ⊂ `供配电工程`）只给 0.45~0.80，
    这类**不该**进人工队列（"供配电"与"配电"是两个工程类别）。
    故此处统一用同一个下限过滤，两处判据不会分叉。

    ★ 与 match_materials 的区别
    --------------------------
    `match_materials` 面向**用户查询**，宁可多给候选（用户自己会看）；
    本函数面向**人工审核队列**，宁可少报——队列一旦被噪声淹没就没人看了。
    故此处要求短名至少 MIN_CONTAIN_LEN 字，2 字短名（`空调`、`设备`）一律不报。
    """
    if a == b:
        return None

    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    cs, cl = (ca, cb) if len(a) <= len(b) else (cb, ca)
    ambiguous = False
    head_modifier = False

    if short in long_:
        parts = _containment_parts(short, long_)
        score = _containment_score(short, long_)
        if parts is None or score <= 0:
            return None
        _short, _long, extra, head, tail = parts
        ambiguous = _containment_is_insertion(short, long_)
        head_modifier = bool(head) and not tail
        reason = "包含匹配"
        method = f"「{short}」完整包含于「{long_}」，多出「{extra}」{len(extra)} 字"
        if ambiguous:
            method += "；该定语可能改变物料类别（需重点核实）"
        elif head_modifier:
            method += "；定语加在**短名之前**，可能限定了类别（请核实是否同一物料）"
    else:
        score = _similarity(a, b)
        reason = "字符重叠"
        method = "无包含关系，但字符相似度极高（疑似错别字 / 同义改写）"

    if score < SUGGEST_MIN_SCORE:
        return None

    # ★ 置信度分档：**"确定性不足"的包含永不进"高"档**
    # ---------------------------------------------
    # `配电工程` ⊂ `供配电工程`（前置"供"）与 `空调` ⊂ `空调配件`（后置"配件"）
    # 在字符串上几乎同形，但"供配电"与"配电"是两个工程类别。
    # 分数只能差出 0.02，靠它分档会误导人工直接执行，
    # 故改用**结构信号**封顶：见 _containment_is_insertion。
    if score >= SUGGEST_HIGH and not ambiguous:
        level = "高"
    elif score >= SUGGEST_MEDIUM:
        level = "中"
    else:
        level = "低"

    total = cs + cl
    return {
        "keep": short,
        "merge": long_,
        "keep_count": cs,
        "merge_count": cl,
        "combined_count": total,
        "score": round(score, 4),
        "match_reason": reason,
        "confidence": level,
        "why": method,
        "ambiguous": ambiguous,
        "head_modifier": head_modifier,
        "transitive_conflict": False,
        "suggestion": f"建议将「{long_}」归并到「{short}」（归并后共 {total} 条记录）",
    }


def suggest_standardization(limit: int = DEFAULT_SUGGEST_LIMIT) -> dict[str, Any]:
    """列出"疑似应归并但尚未归并"的物料对，供人工确认。

    ★ 为什么要做这个接口，而不是把规则调得更激进
    -----------------------------------------
    归一化规则（`normalize_name`）只能处理**有确定性的**后缀。
    但数据里还有大量需要**语义判断**的情况::

        复印机 ~ 复印机配件      —— 配件是不是同一个物料？取决于业务口径
        高标准农田建设 ~ 高标准农田提升建设 —— 是不是同一个标的？
        包装用纸 ~ 包装纸        —— 明显同一物料，但没有可枚举的规则

    把这些硬塞进规则会造成**不可控的误归并**（把"配件"并进"整机"，
    价格统计直接失真）。故本接口只**提出候选 + 给出理由和置信度**，
    由人工确认后再写进规则 —— 与 `analysis.py` 的"标记而非删除"同一思路：
    **系统负责发现，人负责决策**。

    ★ 已知局限：无法区分"大类"与"同一物料"（有意不修）
    ------------------------------------------------
    实测候选里有这类对子::

        羽毛球 ~ 羽毛球拍 / 羽毛球网 / 羽毛球服
        麦克风/话筒 ~ 麦克风/话筒支架 / 麦克风/话筒咪罩

    前者是**大类与具体商品**（球 ≠ 球拍），后者是**主件与配件**。
    二者"该不该并"取决于业务口径，**不是字符串相似度能回答的问题**——
    本接口把它们如实报出来并给出理由，正是为了让业务方来定；
    若强行用规则区分，就会重新引入不可控的误归并。
    故此处只保证"不自动执行"，不保证"候选全对"。

    ★ 为什么用倒排索引做候选筛选
    --------------------------
    2272 个标准物料两两比较是 258 万对，每对算一次 difflib 需要数十秒，接口不可用。
    实测**同一物料的两种写法必然共享字符**，故以字符为 key 建倒排表，
    只为"至少共享 2 个字符、且长度差 ≤6"的对打分——实测候选对落在 5.6 万量级，
    全流程约 1.4s（其中建索引 4ms、筛候选 0.4s、打分 1.0s）。
    这是**人工审核接口**，不是高频查询，秒级可接受；
    若将来要压到百毫秒级，把 `_similarity` 换成预计算的 bigram 倒排检索即可，
    但当前没有必要提前优化。

    返回的每一对含 `keep`（建议保留的短名）、`merge`（建议并入的名）、
    两边各自的交易次数、得分、置信度（高/中/低）、**可读的归并理由**，
    以及 `transitive_conflict`（该对的归并目标本身也被建议归并到别处）。
    """
    n = _clamp(limit, 1, MAX_SUGGEST_LIMIT)
    index = _index()
    counts = index["counts"]
    names = sorted(counts)

    # 倒排索引：字符 → 含该字符的标准物料（跳过 1 字名，它们太泛）
    inverted: dict[str, list[str]] = {}
    for name in names:
        for ch in set(name):
            inverted.setdefault(ch, []).append(name)

    seen: set[tuple[str, str]] = set()
    pairs: list[dict[str, Any]] = []
    for name in names:
        if len(name) < MIN_REMAIN_LEN:
            continue
        # 统计每个候选与 name 共享的字符数
        shared: Counter[str] = Counter()
        for ch in set(name):
            for other in inverted.get(ch, ()):
                if other != name:
                    shared[other] += 1
        for other, common in shared.items():
            # 至少共享 2 个字符才值得算相似度（1 个字符的共性没有信息量）
            if common < 2 or abs(len(other) - len(name)) > 6:
                continue
            pair = (name, other) if name < other else (other, name)
            if pair in seen:
                continue
            seen.add(pair)
            hit = _suggest_pair(name, other, counts[name], counts[other])
            if hit:
                pairs.append(hit)

    # 排序 = 人工审核的优先级，三层：
    #   ① transitive_conflict —— 归并目标本身也被建议归并到别处，队列不能自相矛盾
    #   ② ambiguous / head_modifier —— 定语可能改变类别的（`配电工程`~`供配电工程`）
    #   ③ 分数、总次数 —— 高频、清晰者优先
    #
    # ★ ② 为什么能解决"分数分不开"的问题
    # ---------------------------------
    # `配电工程`⊂`供配电工程`(0.82) 与 `医学检验试剂、耗材`⊂`…耗材等`(0.85) 只差 0.03，
    # 按分数排序时几乎并排。改成按**结构标记**排在前面/后面后，
    # 人工先看到的一律是"定语明确不改变身份"的条目，可疑项沉底且有文字说明。
    # 这比继续拧分数公式更可靠——**结构判据可解释，阈值调参不可解释**。
    keeps = {p["keep"] for p in pairs if p["confidence"] == "高"}
    for p in pairs:
        p["transitive_conflict"] = p["merge"] in keeps and p["keep"] not in keeps

    pairs.sort(
        key=lambda p: (
            p["transitive_conflict"],
            p["ambiguous"] or p["head_modifier"],
            -p["score"],
            -p["combined_count"],
            p["keep"],
            p["merge"],
        )
    )
    top = pairs[:n]
    conflicts = sum(1 for p in pairs if p["transitive_conflict"])
    ambiguous_n = sum(1 for p in pairs if p["ambiguous"] or p["head_modifier"])
    high_n = sum(1 for p in pairs if p["confidence"] == "高")

    return {
        "total_candidates": len(pairs),
        "returned": len(top),
        "high_confidence": high_n,
        "transitive_conflicts": conflicts,
        "ambiguous_pairs": ambiguous_n,
        "items": top,
        "rules": {
            "min_score": SUGGEST_MIN_SCORE,
            "high": SUGGEST_HIGH,
            "medium": SUGGEST_MEDIUM,
            "note": "分数 = 包含关系加权或字符相似度；数量后缀已在归一化阶段自动处理，不占人工队列",
        },
        "notes": [
            "本接口只提出候选，**不自动归并**。确认后请把规则写进 "
            "normalize_name 的后缀表，使归并对所有人可见、可审计",
            f"{high_n} 对为高置信（定语明确不改变物料身份），建议优先处理",
            f"{ambiguous_n} 对标记 ambiguous / head_modifier（定语可能改变类别，"
            "如 `配电工程`~`供配电工程`），已排到队列后段，请重点核实",
            f"{conflicts} 对存在传递性冲突（其归并目标本身也被建议归并到别处），"
            "已排在最后并标 transitive_conflict=1",
        ],
    }


# ---------------------------------------------------------------------------
# ⑩ 对外接口 5/5：归一化预览（调试/规则验证用）
# ---------------------------------------------------------------------------


def normalize_preview(names: list[str]) -> dict[str, Any]:
    """批量预览归一化结果，并标出"会与物料库中哪些写法归并"。

    这个接口不影响业务，但**规则一旦出错影响面很大**（全库口径），
    故给运维/测试一个不改数据就能验证规则的手段。
    """
    index = _index()
    items = []
    for raw in names or []:
        norm = normalize_name(raw)
        group = index["variants"].get(norm)
        items.append(
            {
                "raw": raw,
                "normalized": norm,
                "changed": norm != str(raw).strip().upper(),
                "in_library": bool(group),
                "library_variants": [r for r, _ in (group or [])],
            }
        )
    return {"count": len(items), "items": items}


__all__ = [
    "DEFAULT_MATCH_LIMIT",
    "GENERIC_SUFFIXES",
    "MAX_MATCH_LIMIT",
    "MIN_MATCH_SCORE",
    "MIN_REMAIN_LEN",
    "MaterialError",
    "NOISE_SUFFIXES",
    "SUGGEST_MIN_SCORE",
    "clear_cache",
    "match_materials",
    "material_detail",
    "material_stats",
    "normalize_name",
    "normalize_preview",
    "suggest_standardization",
]
