"""postgresql_client 的白名单校验与故障降级契约测试（全程不连真实数据库）。

为什么这些测试重要
------------------
`src/database/postgresql_client.py` 里有两类"绝不能出错"的地方，它们都不是"查询结果对不对"，
而是"错误输入会不会变成 SQL 的一部分"：

1. **白名单是标识符位置上的唯一防线**（`分配说明.md` 的防注入红线）。SQL 的**值**可以交给
   `%s` 参数化，但**字段名 / 聚合函数名 / GROUP BY 的表达式顶不进占位符**——把
   `purchaser; DROP TABLE bidding_records` 拼到标识符的位置上，参数化救不了。所以
   `validate_field` / `validate_agg` / `trend` 的校验粒度、以及两个白名单集合本身的成员，
   都在这里被钉死：有人"顺手加一个字段"就是扩大攻击面，必须让测试显性变红。
   同一件事的另一半是"用户输入只进参数、不进语句"——本文件用假游标把真正发给驱动的
   SQL 与参数捞出来直接断言，而不是只看返回值。

2. **故障与参数错误要走相反的两条路**：数据库宕机是运营故障 → 返回 `[]` / `-1` /
   `{"ok": False, ...}`，Agent 不能因为后端挂了而崩；而"字段名写错了"是调用方缺陷 →
   显式抛 `ValueError`，静默返回空会让它看起来像"库里没数据"。这两条都被固定在本文件里，
   顺带保证 `health()` 的中文提示不泄露主机名、端口与驱动的原始报文。

不连库的做法：用 `_FakeConnection` / `_FakeCursor` 顶替 `psycopg.connect`。除了让测试脱离
PostgreSQL 服务（CI 里没有库），更重要的是**假连接会记录发出去的语句与参数**——这是
"参数化到底有没有生效"唯一可靠的断言方式。真库联调走 `python main.py api` 与人工验收，
不属于单测范围。

运行：
    uv run pytest tests/test_pg_validation.py -v      # 项目标准写法
    .venv/Scripts/python.exe -m pytest tests/test_pg_validation.py -v   # 本机：不触发 uv.lock 改写
"""

import pytest
import psycopg
from psycopg.types.json import Jsonb

import src.database.postgresql_client as pgmod

# __init__ 缺配置会直接抛 ValueError，而 CI 与没配 .env 的机器上这些变量都不存在，
# 所以每个用例自己预置一份占位配置，不依赖本地 .env（conftest.py 只预置了向量库与 LLM 的键）。
POSTGRES_ENV = {
    "POSTGRES_HOST": "127.0.0.1",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "chatbot",
    "POSTGRES_USER": "postgres",
    "POSTGRES_PASSWORD": "test-password",
}


def _preset_env(monkeypatch) -> None:
    """预置 POSTGRES_*：值本身不重要，重要的是"配置齐全"这个前置条件。"""
    for name, value in POSTGRES_ENV.items():
        monkeypatch.setenv(name, value)


def _make_client(monkeypatch) -> pgmod.PostgresClient:
    _preset_env(monkeypatch)
    return pgmod.PostgresClient()


class _FakeCursor:
    """假游标：记录发出去的 SQL 与参数，并按队列返回结果行。

    记录而不是只返回空，是因为本文件最要紧的断言不是"查到了什么"，
    而是"发给驱动的语句里有没有掺进用户输入"。
    """

    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.executed = []  # [(sql, params), ...]

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeConnection:
    """假连接：上下文语义与 psycopg 一致（进块返回自身，出块不吞异常）。

    `__exit__` 返回 False 很关键——真 psycopg 连接块内抛异常会向上传播，
    假连接如果吞掉异常，"降级"的测试就会变成永远通过的空转。
    """

    def __init__(self, rows=None):
        self.cursor_obj = _FakeCursor(rows)

    def cursor(self):
        return self.cursor_obj

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    @property
    def executed(self):
        """已执行的语句列表，等价于"这次调用真的碰了数据库吗"。"""
        return self.cursor_obj.executed


def _install_fake_connect(monkeypatch, rows=None):
    """把 psycopg.connect 换成假连接，返回假连接对象供断言。

    补丁打在 `pgmod.psycopg.connect` 上：本模块是 `import psycopg` 之后按属性调用
    （`psycopg.connect(...)`），改这一个属性即覆盖全部调用点；monkeypatch 会在用例结束后还原。
    """
    conn = _FakeConnection(rows)
    monkeypatch.setattr(pgmod.psycopg, "connect", lambda **kwargs: conn)
    return conn


def _install_failing_connect(monkeypatch, make_error):
    """把 psycopg.connect 换成必抛的桩：模拟数据库宕机 / 网络不通。

    make_error 传工厂而不是异常实例，避免同一个异常对象被多个用例复用（traceback 会串味）。
    """

    def _connect(**kwargs):
        raise make_error()

    monkeypatch.setattr(pgmod.psycopg, "connect", _connect)


def _install_recording_connect(monkeypatch, rows=None):
    """假连接 + 调用计数：用来钉死"这条路径根本不该连库"。"""
    conn = _FakeConnection(rows)
    calls = []

    def _connect(**kwargs):
        calls.append(kwargs)
        return conn

    monkeypatch.setattr(pgmod.psycopg, "connect", _connect)
    return conn, calls


# ---------------------------------------------------------------- 1. 字段白名单

# 表里真实存在的列不等于"允许出现在 SQL 标识符位置"：`amount` / `created_at` 都是列，
# 但不在分组白名单里 —— 白名单宁可窄，也不留一个能用字符串拼进去的位置。
_ILLEGAL_FIELDS = (
    "id; DROP TABLE bidding_records",
    "amount)--",
    "1=1",
    "purchaser; DROP TABLE bidding_records",
    "purchaser, supplier",
    "amount",
    "created_at",
    "",
    "   ",
    None,
)


def test_validate_field_normalizes_legal_names():
    """合法字段返回规范化值：LLM 抽出来的 `Purchaser`、`supplier ` 与库里的列名只差大小写。"""
    assert pgmod.validate_field("subject_matter") == "subject_matter"
    assert pgmod.validate_field("  Purchaser  ") == "purchaser"
    assert pgmod.validate_field("SUBJECT_MATTER") == "subject_matter"
    assert pgmod.validate_field("Location_City") == "location_city"


@pytest.mark.parametrize("field", _ILLEGAL_FIELDS, ids=[repr(f) for f in _ILLEGAL_FIELDS])
def test_validate_field_rejects_illegal_input(field):
    """非法输入必须抛 ValueError，且报文里带上原值（排查"字段名写错了"时要看得见）。"""
    with pytest.raises(ValueError) as excinfo:
        pgmod.validate_field(field)
    assert "不支持的查询字段" in str(excinfo.value)


def test_group_field_whitelist_members_are_frozen():
    """白名单集合本身是安全边界：多一个成员就多一个注入口子，改动必须在这里显性失败。"""
    assert pgmod.ALLOWED_GROUP_FIELDS == {
        "project_name",
        "subject_matter",
        "purchaser",
        "agency",
        "supplier",
        "location_city",
    }


# ---------------------------------------------------------------- 2. 聚合白名单


@pytest.mark.parametrize("agg", ["count", "sum", "avg", "min", "max"])
def test_validate_agg_accepts_whitelisted_functions(agg):
    assert pgmod.validate_agg(agg) == agg


def test_validate_agg_is_case_insensitive():
    """大小写不敏感且返回规范化小写：调用方（或 LLM）写 `COUNT` 不该被拒。"""
    assert pgmod.validate_agg("COUNT") == "count"
    assert pgmod.validate_agg("  Sum ") == "sum"


@pytest.mark.parametrize("agg", ["drop", "exec", "", "   ", None, "count(*)", "count; DROP TABLE x"])
def test_validate_agg_rejects_non_whitelisted(agg):
    """`count(*)` 这种"看着像聚合"的输入也要拒：白名单比的是函数名，不是像不像。"""
    with pytest.raises(ValueError) as excinfo:
        pgmod.validate_agg(agg)
    assert "不支持的聚合函数" in str(excinfo.value)


def test_agg_whitelist_members_are_frozen():
    assert pgmod.ALLOWED_AGG == {"count", "sum", "avg", "min", "max"}


# ---------------------------------------------------------------- 3. trend 的时间粒度


def test_trend_rejects_bad_group_by_before_touching_db(monkeypatch):
    """非法粒度直接抛，且**发生在连库之前**：这是标识符位置的输入，必须先拒再查。"""
    client = _make_client(monkeypatch)
    _, calls = _install_recording_connect(monkeypatch)

    with pytest.raises(ValueError) as excinfo:
        client.trend(group_by="month; DROP TABLE bidding_records")

    assert "不支持的时间粒度" in str(excinfo.value)
    assert calls == []  # 一个连接都不该开


@pytest.mark.parametrize("raw,key", [("month", "month"), ("year", "year"), ("  YEAR ", "year")])
def test_trend_uses_constant_expression_from_table(raw, key, monkeypatch):
    """合法粒度也不拼字符串：表达式取自本模块的常量表，外部输入只当 key 用。"""
    conn = _install_fake_connect(monkeypatch)
    client = _make_client(monkeypatch)

    client.trend(group_by=raw, limit=5)

    sql, params = conn.executed[0]
    assert pgmod._TREND_EXPRESSIONS[key] in sql
    assert params == (5,)


# ---------------------------------------------------------------- 4. 故障降级（不抛异常）

# 每个公开方法在数据库不可用时的返回值：形状必须与"查到了空结果"可区分
# （`-1` / `False` / `{}` 都是"没办成"，而不是"结果是空"）。
_DEGRADE_CASES = {
    "top_subject": (lambda c: c.top_subject(), []),
    "by_purchaser": (lambda c: c.by_purchaser("某某局"), []),
    "by_supplier": (lambda c: c.by_supplier("某某公司"), []),
    "amount_range": (lambda c: c.amount_range(100, 1000), []),
    "location_dist": (lambda c: c.location_dist(), []),
    "trend": (lambda c: c.trend("month"), []),
    "purchaser_ranking": (lambda c: c.purchaser_ranking(), []),
    "supplier_ranking": (lambda c: c.supplier_ranking(), []),
    "avg_amount_by_subject": (lambda c: c.avg_amount_by_subject("空调"), []),
    "search_by_keyword": (lambda c: c.search_by_keyword("空调"), []),
    "stats_overview": (lambda c: c.stats_overview(), {}),
    "create_conversation": (lambda c: c.create_conversation("测试会话"), -1),
    "list_conversations": (lambda c: c.list_conversations(), []),
    "delete_conversation": (lambda c: c.delete_conversation(1), False),
    "save_feedback": (lambda c: c.save_feedback(1, 1, "up"), False),
    "save_message": (lambda c: c.save_message(1, "user", "你好"), -1),
    "load_messages": (lambda c: c.load_messages(1), []),
    "ensure_schema": (lambda c: c.ensure_schema(), False),
}

# RuntimeError 是"没想到的异常"：降级不能只认 psycopg 自己的异常类型，
# 否则一个编码错误就能把整轮问答带崩（工具层 4 线程并行，异常会穿透到 Agent）。
_CONNECT_ERRORS = (
    ("OperationalError", lambda: psycopg.OperationalError("connection refused")),
    ("RuntimeError", lambda: RuntimeError("unexpected")),
)


@pytest.mark.parametrize("make_error", [e[1] for e in _CONNECT_ERRORS], ids=[e[0] for e in _CONNECT_ERRORS])
@pytest.mark.parametrize("name", sorted(_DEGRADE_CASES))
def test_public_methods_degrade_without_raising(name, make_error, monkeypatch):
    """库不可用时每个公开方法都返回哨兵值，绝不向外抛——Agent 不能因为后端宕机而崩。"""
    run, expected = _DEGRADE_CASES[name]
    client = _make_client(monkeypatch)
    _install_failing_connect(monkeypatch, make_error)

    assert run(client) == expected


# ---------------------------------------------------------------- 5. health 契约


def test_health_reports_failure_without_leaking_internals(monkeypatch):
    """不可达时返回 {"ok": False, "error": ...}；error 只给方向，不含主机、端口与驱动原文。

    这个字符串会进 /api/health 与用户可见区，驱动原始报文里带着主机名、端口、库名，
    泄露出去既没用又难看。
    """
    client = _make_client(monkeypatch)
    _install_failing_connect(
        monkeypatch, lambda: psycopg.OperationalError("connection to server at 127.0.0.1, port 5432 failed")
    )

    result = client.health()

    assert result["ok"] is False
    assert "POSTGRES_*" in result["error"]  # 指向该检查的配置组
    assert "127.0.0.1" not in result["error"]
    assert "5432" not in result["error"]
    assert "connection to server" not in result["error"]


def test_health_hides_unexpected_exception_details(monkeypatch):
    """非连接类异常也走同一条降级路径，同样不把原始报文透出去。"""
    client = _make_client(monkeypatch)
    _install_failing_connect(monkeypatch, lambda: RuntimeError("secret internal detail"))

    result = client.health()

    assert result["ok"] is False
    assert "secret internal detail" not in result["error"]
    assert "详见后端日志" in result["error"]


def test_health_ok_on_reachable_db(monkeypatch):
    """连通时 error 必须是 None（不是空串）：调用方靠 `is None` 判断是否真成功过。"""
    _install_fake_connect(monkeypatch, rows=[{"?column?": 1}])
    client = _make_client(monkeypatch)

    assert client.health() == {"ok": True, "error": None}


# ---------------------------------------------------------------- 6. 空输入 / 参数化 / 通配符


@pytest.mark.parametrize("keyword", ["", "   ", None])
def test_search_by_keyword_empty_short_circuits_without_query(keyword, monkeypatch):
    """空关键词直接返回 []，连库都不连：空串会变成 ILIKE '%%'，那是全表匹配。"""
    client = _make_client(monkeypatch)
    _, calls = _install_recording_connect(monkeypatch)

    assert client.search_by_keyword(keyword) == []
    assert calls == []


@pytest.mark.parametrize("subject", ["", "  ", None])
def test_avg_amount_by_subject_empty_short_circuits_without_query(subject, monkeypatch):
    """同上：空标的物关键词不该变成一次全表 AVG。"""
    client = _make_client(monkeypatch)
    _, calls = _install_recording_connect(monkeypatch)

    assert client.avg_amount_by_subject(subject) == []
    assert calls == []


def test_search_by_keyword_passes_input_as_parameter_not_sql(monkeypatch):
    """防注入红线：用户输入一个字符都不进语句，只作为 ILIKE 的模式串参数。

    这条断言是"参数化真的生效"的实证——白名单管的是标识符，值的安全性全靠这里。
    """
    conn = _install_fake_connect(monkeypatch, rows=[])
    client = _make_client(monkeypatch)
    attack = "'; DROP TABLE bidding_records; --"

    client.search_by_keyword(attack)

    sql, params = conn.executed[0]
    assert attack not in sql
    assert "DROP" not in sql.upper()
    assert len(params) == 5  # 四个字段各一个占位符 + limit
    # 连表名里的 `_` 都被转义了（它是 LIKE 的单字符通配符）：模式串是转义后的形态，
    # 而不是原文 —— 转义发生在 Python 侧、注入防护交给占位符，两者各管一段
    assert params[:4] == ("%'; DROP TABLE bidding\\_records; --%",) * 4
    assert params[4] == 20  # 默认 limit


def test_search_by_keyword_escapes_like_wildcards(monkeypatch):
    """参数化管注入，管不了通配符语义：`%` / `_` 必须转义后再交给 ILIKE。

    不转义时用户搜一个 `%` 就能匹配全表——返回一堆不相干记录却不报错，比注入更难发现。
    """
    conn = _install_fake_connect(monkeypatch, rows=[])
    client = _make_client(monkeypatch)

    client.search_by_keyword("100%")

    assert conn.executed[0][1][0] == "%100\\%%"


def test_by_purchaser_passes_value_as_parameter(monkeypatch):
    """精确匹配走 `= %s`：同样只有参数、没有拼接。"""
    conn = _install_fake_connect(monkeypatch, rows=[])
    client = _make_client(monkeypatch)

    client.by_purchaser("某某局'; DROP TABLE bidding_records; --")

    sql, params = conn.executed[0]
    assert "DROP" not in sql.upper()
    assert params[0] == "某某局'; DROP TABLE bidding_records; --"


# ---------------------------------------------------------------- 7. 数值参数与配置兜底


@pytest.mark.parametrize("limit,expected", [(3, 3), (0, 1), (-5, 1), (999999, pgmod.MAX_LIMIT), ("abc", 10), (None, 10)])
def test_limit_is_clamped_to_safe_range(limit, expected, monkeypatch):
    """limit 多由 LLM 给出：非法值退回默认、超上限截断，别让 `LIMIT %s` 收到字符串。"""
    conn = _install_fake_connect(monkeypatch)
    client = _make_client(monkeypatch)

    client.top_subject(limit=limit)

    assert conn.executed[0][1] == (expected,)


def test_save_message_accepts_none_sources(monkeypatch):
    """sources=None 要落成 SQL NULL（"没有来源"），不是 JSON 的 null（"字段存在且为 null"）。"""
    conn = _install_fake_connect(monkeypatch, rows=[{"id": 7}])
    client = _make_client(monkeypatch)

    assert client.save_message(1, "user", "你好", sources=None) == 7

    params = conn.executed[0][1]
    assert params[3] is None
    assert params[4] is None


def test_save_message_wraps_json_sources_with_jsonb(monkeypatch):
    """dict / list 必须包成 Jsonb：JSONB 列收到裸 dict，psycopg 会当普通对象适配而报错。"""
    conn = _install_fake_connect(monkeypatch, rows=[{"id": 8}])
    client = _make_client(monkeypatch)

    assert client.save_message(1, "assistant", "答案", sources=[{"tool": "web_search"}]) == 8

    payload = conn.executed[0][1][3]
    assert isinstance(payload, Jsonb)
    # 断言包里裹的是原始 Python 结构（Jsonb.obj），而不是已经序列化好的字符串——
    # 传 json.dumps(...) 的字符串也能入库，但那样存进去是 JSON 字符串而非 JSON 对象
    assert payload.obj == [{"tool": "web_search"}]


def test_save_message_returns_minus_one_when_insert_fails(monkeypatch):
    """写入没回 id（表不存在 / 外键不过）时返回 -1，不返回 0——SERIAL 从 1 开始，0 不是合法 id。"""
    _install_fake_connect(monkeypatch, rows=[])
    client = _make_client(monkeypatch)

    assert client.save_message(1, "user", "你好") == -1


def test_client_requires_postgres_env(monkeypatch):
    """缺必需配置时抛 ValueError 并点名缺了哪个——静默用一个空主机名去连，报错会离病因很远。"""
    for name in ("POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError) as excinfo:
        pgmod.PostgresClient()

    assert "POSTGRES_DB" in str(excinfo.value)


def test_port_falls_back_to_default(monkeypatch):
    """端口缺失或写成非数字都退 5432：它有众所周知的默认值，不值得让整台机器起不来。"""
    _preset_env(monkeypatch)

    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    assert pgmod.PostgresClient().port == 5432

    monkeypatch.setenv("POSTGRES_PORT", "not-a-number")
    assert pgmod.PostgresClient().port == 5432

    monkeypatch.setenv("POSTGRES_PORT", "6543")
    assert pgmod.PostgresClient().port == 6543
