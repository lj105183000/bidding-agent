"""测试数据：DSML 风格工具调用的泄漏样本与变体。

★ 本模块为什么单独存在
==================
被测标记含**全角竖线** U+FF5C。该字符原样写进源码/工具调用参数会让
上游序列化出问题（实测踩到两次），故一律用 chr(0xFF5C) 运行时拼装，
文件里不出现该字符。

★ 这些样本从哪来
============
主样本 `USER_SAMPLE` 是**用户实际遇到的泄漏输出**（原样保留结构与内容），
其余变体来自"真实可能出现的写法"：不同网关的 ASCII/全角混用、
命名空间前缀、截断未闭合、单引号属性等。
"""

from __future__ import annotations

FW = chr(0xFF5C)  # 全角竖线 U+FF5C
MARK = f"{FW}{FW}DSML{FW}{FW}"  # 主样本里的标记形态

#: 用户实际遇到的泄漏样本（两个并行 invoke，keyword 分别是"管理局"/"中学"）
USER_SAMPLE = f"""<{MARK}tool_calls>
<{MARK}invoke name="search_knowledge_graph">
<{MARK}parameter name="keyword" string="true">管理局</parameter>
<{MARK}parameter name="query_type" string="true">search_entity</parameter>
</invoke>
<{MARK}invoke name="search_knowledge_graph">
<{MARK}parameter name="keyword" string="true">中学</parameter>
<{MARK}parameter name="query_type" string="true">search_entity</parameter>
</invoke>
</{MARK}tool_calls>"""

#: 单层全角标记
FW_SINGLE = (
    f'<{FW}DSML{FW}invoke name="search_knowledge_base">'
    f'<{FW}DSML{FW}parameter name="query">投标保证金</parameter></invoke>'
)

#: ASCII 竖线（部分网关会把它降级成 ASCII）
ASCII_BAR = (
    '<||DSML||invoke name="search_knowledge_base">'
    '<||DSML||parameter name="query">投标保证金</parameter></invoke>'
)

#: 没有任何 DSML 前缀的纯 XML
PLAIN_XML = (
    '<invoke name="search_knowledge_base">'
    '<parameter name="query">投标保证金</parameter></invoke>'
)

#: 小写 dsml（大小写不敏感）
LOWER_DSML = (
    f'<{FW}dsml{FW}invoke name="search_knowledge_base">'
    f'<{FW}dsml{FW}parameter name="query">投标保证金</parameter></invoke>'
)

#: 命名空间前缀
NS_PREFIX = (
    '<ns:invoke name="search_knowledge_base">'
    '<ns:parameter name="query">投标保证金</ns:parameter></ns:invoke>'
)

#: 单引号属性
SINGLE_QUOTE = (
    "<invoke name='search_knowledge_base'>"
    "<parameter name='query'>投标保证金</parameter></invoke>"
)

#: invoke 被截断、没有收尾标签（真实场景：模型输出被 max_tokens 截掉）
TRUNCATED = (
    f'<{MARK}invoke name="search_knowledge_graph">\n'
    f'<{MARK}parameter name="keyword">管理局</parameter>'
)

#: 多参数
MULTI_PARAM = (
    '<invoke name="query_database">'
    '<parameter name="sql">SELECT 1</parameter>'
    '<parameter name="limit">10</parameter></invoke>'
)

#: 全部应被识别为工具调用的变体：label -> (文本, 期望调用数, 期望首个工具名)
VARIANTS: list[tuple[str, str, int, str]] = [
    ("全角竖线单层", FW_SINGLE, 1, "search_knowledge_base"),
    ("ASCII 竖线", ASCII_BAR, 1, "search_knowledge_base"),
    ("纯 XML 无前缀", PLAIN_XML, 1, "search_knowledge_base"),
    ("小写 dsml", LOWER_DSML, 1, "search_knowledge_base"),
    ("命名空间前缀", NS_PREFIX, 1, "search_knowledge_base"),
    ("单引号属性", SINGLE_QUOTE, 1, "search_knowledge_base"),
    ("invoke 未闭合", TRUNCATED, 1, "search_knowledge_graph"),
    ("多参数", MULTI_PARAM, 1, "query_database"),
]

#: 不该被判为工具调用的正常文本（防误伤）
BENIGN: list[str] = [
    "可以使用 search_knowledge_base 工具来检索知识库。",
    "根据《政府采购法》第三十一条，单一来源采购适用三种情形。",
    "函数 f(x) 的定义域是实数集。",
    "查询类型有 search_entity 与 search_relation 两种。",
    "中标金额为 120 万元，工期 180 天。",
]
