"""同轮工具**并行执行 + 同参去重**的回归测试。

★ 为什么这些用例必须存在（这次改的是编排层最容易出错的地方）
=====================================================
工具从"逐个 await"改成"同轮并发"后，有四类错误**不会报错、只会悄悄出错**：

    ① 结果串位：并发完成次序 ≠ 决策次序。若按完成次序回填，
       模型会把 A 的结果当成 B 的结果（引用的法条对不上号）
    ② 观测值张冠李戴：RAG 耗时分解/图表原本写在**实例属性**上，
       那是"一次只跑一个工具"才成立的通道。并发时谁最后写完就是谁的值
    ③ 去重把指标洗白：去重后若不再计入 `tool_call_names`，
       评测里的"重复调用次数"会因为我们去重而假性变好
    ④ 打转提示失准：计数少算 → 模型在一个坑里反复检索却收不到提示

故这里逐条钉住：顺序、计时归属、去重计数、超时隔离、异常隔离。
"""

import json
import time

import pytest

import src.agent.core as core_mod
import src.agent.react_loop as loop_mod
from src.agent import BiddingAgent


class _ScriptedLLM:
    """按脚本返回 chat_raw；chat_stream 返回固定分片。"""

    def __init__(self, raw_responses=(), stream_chunks=("答",)):
        self.raw_responses = list(raw_responses)
        self.stream_chunks = list(stream_chunks)
        self.raw_calls = 0

    def chat_raw(self, messages, tools=None):
        self.raw_calls += 1
        item = self.raw_responses.pop(0) if self.raw_responses else "直接回答"
        if isinstance(item, Exception):
            raise item
        return item

    def chat_stream(self, messages):
        yield from self.stream_chunks


def _install(monkeypatch, mapping, log=None):
    """替换两处 TOOL_EXECUTORS 绑定并放开白名单。"""
    def make(name, fn):
        def wrapper(arguments, question=""):
            if log is not None:
                log.append((name, arguments))
            return fn(arguments)
        return wrapper

    executors = {name: make(name, fn) for name, fn in mapping.items()}
    monkeypatch.setattr(core_mod, "TOOL_EXECUTORS", executors)
    monkeypatch.setattr(loop_mod, "TOOL_EXECUTORS", executors)
    monkeypatch.setattr(
        loop_mod.ReActMixin, "_allowed_tool_names", staticmethod(lambda meta: set(mapping))
    )
    return executors


def _decision(*calls) -> str:
    return "\n".join(json.dumps({"name": n, "arguments": a}) for n, a in calls)


def _done(events):
    done = [e for e in events if e[0] == "done"]
    assert len(done) == 1, f"期望恰好一个 done 事件，实际 {len(done)}"
    return done[0][1]


# ---------------------------------------------------------------------------
# ① 顺序：并发完成次序不影响回填次序
# ---------------------------------------------------------------------------


def test_result_order_follows_decision_order_even_if_completion_is_reversed(monkeypatch):
    """★ 慢的先发、快的后发：回填顺序仍必须是决策顺序。

    构造方式：第一个工具睡 0.15s，第二个立即返回 —— 完成次序必然是
    B、A 颠倒。若实现按"谁先完成谁先回填"，这个用例就会抓到。
    """
    def slow(args):
        time.sleep(0.15)
        return "A 结果", []

    _install(monkeypatch, {"tool_a": slow, "tool_b": lambda args: ("B 结果", [])})
    agent = BiddingAgent(llm_client=_ScriptedLLM(
        raw_responses=[_decision(("tool_a", {"q": "1"}), ("tool_b", {"q": "2"})), "好的"]
    ))

    done = _done(list(agent._chat_events("查一下")))

    assert [t["name"] for t in done["tool_timings"]] == ["tool_a", "tool_b"]
    assert done["tool_names"] == ["tool_a", "tool_b"]
    # 回填给模型的消息里，A 在 B 之前
    fed = [m for m in done.get("_messages_for_final") or []]
    if fed:  # 不同 fake 实现下该键可能不存在，存在时顺带校验
        joined = "\n".join(str(m.get("content", "")) for m in fed)
        assert joined.index("[工具 tool_a 结果]") < joined.index("[工具 tool_b 结果]")


def test_parallel_round_is_actually_faster_than_sum(monkeypatch):
    """★ 并行的**实质**：一轮两个各睡 0.2s 的工具，总耗时应接近 0.2s 而非 0.4s。"""
    def nap(args):
        time.sleep(0.2)
        return "结果", []

    _install(monkeypatch, {"tool_a": nap, "tool_b": nap})
    agent = BiddingAgent(llm_client=_ScriptedLLM(
        raw_responses=[_decision(("tool_a", {}), ("tool_b", {})), "好的"]
    ))

    events = list(agent._chat_events("查一下"))
    done = _done(events)
    phase = {p[0]: p[1] for p in done["phase_times"]}
    tool_phase = phase["检索与搜索"]

    # 串行会是 ~400ms；给足调度余量，只要明显小于 400 即证明并行
    assert tool_phase < 360, f"工具阶段 {tool_phase}ms 看起来仍是串行（应接近 200ms）"
    assert tool_phase >= 150, f"工具阶段 {tool_phase}ms 过短，可能没真正执行"
    # 每个工具各自记自己的耗时，不应都记成整批的耗时
    assert all(t["ms"] < 360 for t in done["tool_timings"]), done["tool_timings"]


# ---------------------------------------------------------------------------
# ② 观测值归属：RAG 计时不能张冠李戴
# ---------------------------------------------------------------------------


def test_rag_timing_is_attached_to_the_right_tool(monkeypatch):
    """★ 两个工具同时产生 RAG 计时，各自必须拿到自己的那份。

    这是"实例属性通道在并发下会被覆盖"的直接回归测试：原实现把
    `take_marks()` 的结果写在 `self.last_tool_rag_timing`，并发跑两个检索
    工具时后写完的会覆盖前一个 —— 用户点开"检索知识库 1.2s"看到的
    可能是另一个工具的分解。

    做法：把 `take_marks()` 换成**按线程返回**的替身（每个工具在自己的
    子线程里调用它），执行器把标记放进线程局部变量。
    """
    import threading

    from src.rag import timing as rag_timing

    slot = threading.local()

    def fake_take_marks():
        return {"total_ms": 1, "steps": [], "owner": getattr(slot, "tag", "?")}

    monkeypatch.setattr(rag_timing, "take_marks", fake_take_marks)

    def make_executor(tag):
        def executor(arguments, question=""):
            slot.tag = tag
            time.sleep(0.05)
            return f"{tag} 文本", []

        return executor

    _install(monkeypatch, {"tool_a": make_executor("A"), "tool_b": make_executor("B")})
    agent = BiddingAgent(llm_client=_ScriptedLLM())

    batch = agent._execute_tool_batch(
        [{"name": "tool_a", "arguments": {"q": "1"}},
         {"name": "tool_b", "arguments": {"q": "2"}}],
        {"tool_a", "tool_b"}, "问题",
    )

    assert [row[0] for row in batch] == ["A 文本", "B 文本"]
    owners = [row[3].get("rag", {}).get("owner") for row in batch]
    assert owners == ["A", "B"], (
        f"RAG 计时张冠李戴：期望 ['A','B']，实际 {owners} "
        "（并发下观测值必须随结果返回，不能写在共享属性上）"
    )
    assert all(row[3].get("ms", 0) < 300 for row in batch)


def test_charts_come_from_the_owning_tool(monkeypatch):
    """图表透传与计时同理：并发时不能互相覆盖。"""
    _install(monkeypatch, {
        "tool_a": lambda args: ("A", [], [{"type": "bar", "name": "A图"}]),
        "tool_b": lambda args: ("B", [], [{"type": "bar", "name": "B图"}]),
    })
    agent = BiddingAgent(llm_client=_ScriptedLLM(
        raw_responses=[_decision(("tool_a", {}), ("tool_b", {})), "好的"]
    ))

    done = _done(list(agent._chat_events("查一下")))
    names = [c.get("name") for c in done.get("charts") or []]
    assert names == ["A图", "B图"], f"图表应按决策顺序来自各自的工具，实际 {names}"


# ---------------------------------------------------------------------------
# ③ 去重（★ JSON 形态在**解析层**已去重，真正需要兜底的是 DSML 形态）
# ---------------------------------------------------------------------------


def _dsml(*calls) -> str:
    """DSML/XML 形态的工具调用文本（本项目有实测泄漏样本与专门测试）。"""
    out = []
    for name, args in calls:
        params = "".join(
            f'<parameter name="{k}" string="true">{v}</parameter>' for k, v in args.items()
        )
        out.append(f'<invoke name="{name}">{params}</invoke>')
    return "\n".join(out)


def test_parser_dedupes_json_duplicates(monkeypatch):
    """JSON 形态：**解析层**已去重，批次只见一条调用。"""
    log = []
    _install(monkeypatch, {"tool_a": lambda args: ("A 结果", [])}, log=log)
    agent = BiddingAgent(llm_client=_ScriptedLLM(
        raw_responses=[_decision(("tool_a", {"q": "同"}), ("tool_a", {"q": "同"})), "好的"]
    ))

    done = _done(list(agent._chat_events("查一下")))

    assert len(log) == 1
    assert done["tool_names"] == ["tool_a"]   # 解析层就只剩一条
    assert done["tool_deduped"] == 0          # 兜底层无需出手


def test_parser_dedupes_dsml_duplicates(monkeypatch):
    """DSML 形态：**解析层也**已去重（`_parse_xml_tool_calls` 末尾同样有去重）。

    ★ 这条用例记录的是一次**结论更正**：我最初以为 DSML 分支在去重之前
    就 `return`、会漏掉重复调用，实测证明它自己也有同名同参去重
    （与 JSON 分支是同一段逻辑）。故这里断言的是"解析层去重"这个事实，
    而不是"批次层兜住了"。
    """
    log = []
    _install(monkeypatch, {"tool_a": lambda args: ("A 结果", [])}, log=log)
    agent = BiddingAgent(llm_client=_ScriptedLLM(
        raw_responses=[_dsml(("tool_a", {"q": "同"}), ("tool_a", {"q": "同"})), "好的"]
    ))

    done = _done(list(agent._chat_events("查一下")))

    assert len(log) == 1
    assert done["tool_names"] == ["tool_a"]
    assert done["tool_deduped"] == 0


def test_batch_dedupe_guard_works_on_a_hand_built_duplicate_list(monkeypatch):
    """★ 兜底层本身有效：**直接**给 `_execute_tool_batch` 两条相同调用
    （绕过解析器），也必须只执行一次。

    为什么要单独测：正常链路上解析层已经去重，兜底层永远不会被触发
    （上面两条用例的 `tool_deduped == 0` 正说明这一点）。若只测链路，
    这段兜底代码等于没有覆盖 —— 而它的存在意义恰恰是"上游万一漏了"。
    """
    log = []
    _install(monkeypatch, {"tool_a": lambda args: ("A 结果", [])}, log=log)
    agent = BiddingAgent(llm_client=_ScriptedLLM())

    rows = agent._execute_tool_batch(
        [{"name": "tool_a", "arguments": {"q": "同"}},
         {"name": "tool_a", "arguments": {"q": "同"}}],
        {"tool_a"}, "问题",
    )

    assert len(log) == 1, "兜底层没起作用：同一个调用执行了两遍"
    assert len(rows) == 2, "结果条数必须与调用条数一致（模型要能看到两条回填）"
    assert rows[0][4] is False and rows[1][4] is True, "第二条应被标记为去重命中"
    assert rows[0][0] == rows[1][0] == "A 结果", "重复项应复用首次结果"


def test_batch_dedupe_normalizes_arg_key_order(monkeypatch):
    """兜底层按**规范化参数**判重：键序不同视为同一个调用。"""
    log = []
    _install(monkeypatch, {"tool_a": lambda args: ("A", [])}, log=log)
    agent = BiddingAgent(llm_client=_ScriptedLLM())

    rows = agent._execute_tool_batch(
        [{"name": "tool_a", "arguments": {"x": "1", "y": "2"}},
         {"name": "tool_a", "arguments": {"y": "2", "x": "1"}}],
        {"tool_a"}, "问题",
    )

    assert len(log) == 1
    assert rows[1][4] is True


def test_different_args_are_not_deduped(monkeypatch):
    """参数不同就是两次真实检索，绝不能合并。"""
    log = []
    _install(monkeypatch, {"tool_a": lambda args: (f"结果{args.get('q')}", [])}, log=log)
    agent = BiddingAgent(llm_client=_ScriptedLLM(
        raw_responses=[_decision(("tool_a", {"q": "1"}), ("tool_a", {"q": "2"})), "好的"]
    ))

    done = _done(list(agent._chat_events("查一下")))

    assert len(log) == 2
    assert done["tool_deduped"] == 0
    assert done["tool_names"] == ["tool_a", "tool_a"]


# ---------------------------------------------------------------------------
# ④ 失败隔离
# ---------------------------------------------------------------------------


def test_one_tool_failing_does_not_affect_the_other(monkeypatch):
    """一个工具抛异常，另一个照常返回（并发下的异常必须被隔离）。"""
    def boom(args):
        raise RuntimeError("工具炸了")

    _install(monkeypatch, {"tool_a": boom, "tool_b": lambda args: ("B 结果", [])})
    agent = BiddingAgent(llm_client=_ScriptedLLM(
        raw_responses=[_decision(("tool_a", {}), ("tool_b", {})), "好的"]
    ))

    done = _done(list(agent._chat_events("查一下")))

    assert done["tool_called"] is True
    names = [t["name"] for t in done["tool_timings"]]
    assert names == ["tool_a", "tool_b"]
    oks = {t["name"]: t["ok"] for t in done["tool_timings"]}
    assert oks == {"tool_a": False, "tool_b": True}
    # 失败的工具不应污染来源
    assert done["sources"] == []


def test_timeout_of_one_tool_does_not_block_the_batch(monkeypatch):
    """一个工具卡住（超过单工具超时），整批仍在超时内返回，其余结果完好。

    用 monkeypatch 把单工具超时压到 1s、慢工具睡 3s：串行实现会让整轮
    起码等 3s；并行 + 每工具超时则应 ≈1s 返回，且慢工具降级为失败文案。
    """
    monkeypatch.setattr(loop_mod, "TOOL_TIMEOUT_SECONDS", 1)

    def hang(args):
        time.sleep(3)
        return "慢结果", []

    _install(monkeypatch, {"tool_a": hang, "tool_b": lambda args: ("B 结果", [])})
    agent = BiddingAgent(llm_client=_ScriptedLLM(
        raw_responses=[_decision(("tool_a", {}), ("tool_b", {})), "好的"]
    ))

    t0 = time.monotonic()
    events = list(agent._chat_events("查一下"))
    elapsed = time.monotonic() - t0
    done = _done(events)

    assert elapsed < 2.6, f"整批耗时 {elapsed:.2f}s，超时工具把整轮拖住了"
    oks = {t["name"]: t["ok"] for t in done["tool_timings"]}
    assert oks == {"tool_a": False, "tool_b": True}


def test_single_call_takes_the_serial_fast_path(monkeypatch):
    """只有一个调用时不走线程池（省掉线程与上下文开销），行为不变。"""
    _install(monkeypatch, {"tool_a": lambda args: ("A 结果", [])})
    agent = BiddingAgent(llm_client=_ScriptedLLM(
        raw_responses=[_decision(("tool_a", {})), "好的"]
    ))

    done = _done(list(agent._chat_events("查一下")))

    assert done["tool_called"] is True
    assert [t["name"] for t in done["tool_timings"]] == ["tool_a"]


def test_whitelist_still_rejects_inside_the_batch(monkeypatch):
    """白名单校验在并行批次里同样生效（安全边界不能因并发而绕过）。"""
    log = []
    _install(monkeypatch, {"tool_a": lambda args: ("A", [])}, log=log)
    # 只允许 tool_a，但模型请求 tool_b
    monkeypatch.setattr(
        loop_mod.ReActMixin, "_allowed_tool_names", staticmethod(lambda meta: {"tool_a"})
    )
    agent = BiddingAgent(llm_client=_ScriptedLLM(
        raw_responses=[_decision(("tool_b", {})), "好的"]
    ))

    done = _done(list(agent._chat_events("查一下")))

    assert log == [], "未暴露的工具被执行了 —— 白名单失效"
    assert done["tool_timings"][0]["ok"] is False


@pytest.mark.parametrize("workers", [1, 4])
def test_batch_works_with_any_parallelism(monkeypatch, workers):
    """并行度 1（退化为串行）与 4 都要给出同样的结果顺序。"""
    monkeypatch.setattr(loop_mod, "TOOL_PARALLEL_WORKERS", workers)
    _install(monkeypatch, {"tool_a": lambda args: ("A", []), "tool_b": lambda args: ("B", [])})
    agent = BiddingAgent(llm_client=_ScriptedLLM(
        raw_responses=[_decision(("tool_a", {}), ("tool_b", {})), "好的"]
    ))

    done = _done(list(agent._chat_events("查一下")))
    assert [t["name"] for t in done["tool_timings"]] == ["tool_a", "tool_b"]
