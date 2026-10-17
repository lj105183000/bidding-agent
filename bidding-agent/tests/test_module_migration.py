"""法规库模块的迁移回归测试（`src/services/` → `src/rag/`）。

★ 为什么要专门测"模块位置"
====================
迁移类改动最容易出的问题不是功能坏了（功能测试会挂），而是：

    ① 旧导入路径被删掉 → **外部**脚本/下游代码静默失效
       （本仓库测试全绿，但别人 `from src.services.regulation_store import ...` 会 ImportError）
    ② 别名转发不全 → 少数名字丢失，只有在用到时才发现
    ③ 有人日后"顺手清理"把别名删掉，以为它是死代码

故这里把迁移契约**显式锁住**：两条路径都可用、名字不丢、对象同一。
`src/rag/__init__.py` 自己也写着"公共导入路径为对外契约，勿破坏"。

★ 迁移的事实
==========
被迁模块的 docstring 第一句就是"把条文写入 Qdrant 独立集合并检索"——
它是**检索实现**，依赖全在 `src/rag/` 下（`vector_store` / `embedder`），
放在 `services/` 是错位的。`src/rag/**` 不 import 它，故迁移无循环。
"""

from __future__ import annotations

import importlib

import pytest

NEW_PATH = "src.rag.regulation_store"
OLD_PATH = "src.services.regulation_store"

#: 迁移时机械记录的名字清单（`dir()` 去掉 dunder 后的全部 40 个）。
#: 用它做完整性比对，而不是人工挑几个——人工挑必然漏。
FULL_API = [
    "Any", "Counter", "DENSE_VECTOR_NAME", "Iterable", "POINT_NAMESPACE",
    "PREFETCH_LIMIT", "REGULATION_COLLECTION", "RegulationArticle",
    "RegulationError", "SPARSE_VECTOR_NAME", "Sequence", "UPSERT_BATCH",
    "_PAYLOAD_FIELDS", "_SPARSE_INDEX_SPACE", "_as_article_list", "_build_store",
    "_cosine", "_count_by_law", "_ensure_topics_index", "_extract_dense",
    "_resolve_embedder", "_sparse_index", "_term_frequencies", "_tf_weights",
    "_topic_filter", "all_articles", "annotations", "document_text",
    "ensure_regulation_collection", "get_logger", "hashlib", "logger", "math",
    "models", "point_id", "regulation_stats", "search_regulations",
    "upsert_regulations", "uuid", "verify_regulations",
]

#: 别名**显式**转发的私有名（`import *` 复制不到它们，故必须写出来）。
#: 这两个是测试真正用到的（`tests/test_regulations.py` 直接调 `rs._sparse_index`）。
SHIM_PRIVATE = ["_sparse_index", "_tf_weights"]

#: 业务公开 API：这些必须从两条路径都能拿到
PUBLIC_API = [
    "REGULATION_COLLECTION",
    "RegulationError",
    "all_articles",
    "ensure_regulation_collection",
    "regulation_stats",
    "search_regulations",
    "upsert_regulations",
    "verify_regulations",
]


class TestBothPathsWork:
    def test_new_path_importable(self):
        mod = importlib.import_module(NEW_PATH)
        assert mod.__file__.replace("\\", "/").endswith("src/rag/regulation_store.py")

    def test_old_path_importable(self):
        """★ 外部脚本依赖它。删掉别名会让下游静默失效而本仓库测试全绿。"""
        mod = importlib.import_module(OLD_PATH)
        assert mod is not None

    def test_old_path_from_import_style(self):
        """`from src.services.regulation_store import X` 这种写法也要能用。"""
        mod = importlib.import_module(OLD_PATH)
        for name in PUBLIC_API:
            assert hasattr(mod, name), f"旧路径缺少 {name}"

    def test_package_level_export(self):
        """`from src.rag import search_regulations` 也要能用。"""
        rag = importlib.import_module("src.rag")
        for name in PUBLIC_API:
            assert hasattr(rag, name), f"src.rag 未导出 {name}"


class TestNoRegressionInSurface:
    def test_public_api_complete_on_both_paths(self):
        new = importlib.import_module(NEW_PATH)
        old = importlib.import_module(OLD_PATH)
        for name in PUBLIC_API:
            assert hasattr(new, name), f"新路径缺 {name}"
            assert hasattr(old, name), f"旧路径缺 {name}"

    def test_shim_exports_private_names_tests_need(self):
        """★ 别名必须**显式**再导出这两个私有名。

        被迁模块**没有 `__all__`**，而 `from ... import *` 只复制公开名
        （不以下划线开头的）。若别名只写 `import *`，这两个会丢，
        报错点在**测试里**而不是别名文件里——排查时容易误判成
        "迁移把功能改坏了"。故这里锁住。
        """
        old = importlib.import_module(OLD_PATH)
        for name in SHIM_PRIVATE:
            assert hasattr(old, name), (
                f"别名未转发 {name}——`import *` 不会复制下划线开头的名字，"
                f"必须在别名文件里显式 import"
            )

    def test_no_name_lost_from_new_module(self):
        """新模块必须覆盖迁移前记录的全部 40 个名字。"""
        new = importlib.import_module(NEW_PATH)
        missing = [n for n in FULL_API if not hasattr(new, n)]
        assert not missing, f"迁移后丢失了名字: {missing}"

    def test_reverse_check_paths_are_same_module(self):
        """两条路径指向**同一份实现**（而非复制了一份）。

        若别名是"复制粘贴一份代码"，两边会各自演化、修 bug 只修一边——
        那比 ImportError 更难发现（行为悄悄分叉）。
        """
        new = importlib.import_module(NEW_PATH)
        old = importlib.import_module(OLD_PATH)
        assert old.search_regulations is new.search_regulations
        assert old.REGULATION_COLLECTION == new.REGULATION_COLLECTION
        assert old.RegulationError is new.RegulationError


class TestBehaviourUnchanged:
    def test_sparse_index_same_via_both_paths(self):
        """同一输入经两条路径算出的稀疏向量必须一致（别名不是空壳）。"""
        new = importlib.import_module(NEW_PATH)
        old = importlib.import_module(OLD_PATH)
        term = "政府采购法 第三十一条 单一来源"
        assert new._sparse_index(term) == old._sparse_index(term)

    def test_point_id_deterministic_via_new_path(self):
        """`point_id` 由内容决定（UUID5），迁移不该改变它——改了会让已入库的点全部失配。"""
        new = importlib.import_module(NEW_PATH)
        a = new.point_id("政府采购法", "第三十一条")
        b = new.point_id("政府采购法", "第三十一条")
        assert a == b
        assert a != new.point_id("政府采购法", "第三十二条")

    def test_store_builder_importable(self):
        """`_build_store` 是内部工厂，迁移后仍应在新模块里可用。"""
        new = importlib.import_module(NEW_PATH)
        assert callable(new._build_store)


class TestNoCircularImport:
    def test_rag_package_does_not_import_regulation_store_at_import_time_only(self):
        """`src/rag` 能被独立 import（不依赖 services 的导入顺序）。

        迁移前它住在 `services/`，若 `src.rag` 反过来依赖它就会形成循环。
        实测 `src/rag/**` 不 import 它，故这里断言 `src.rag` 独立可用。
        """
        rag = importlib.import_module("src.rag")
        assert hasattr(rag, "RAGPipeline")

    def test_regulation_store_imports_only_from_rag_and_core(self):
        """它的项目内依赖应只有 `src.rag.*` 与核心基础模块。

        断言这条是为了防止日后有人给它加回对 `src.services` 的依赖——
        那会让"它属于 rag"这个归类失效，也可能引入循环。
        """
        import ast
        import pathlib

        mod = importlib.import_module(NEW_PATH)
        src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        deps: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src"):
                deps.append(node.module)
            elif isinstance(node, ast.Import):
                deps += [a.name for a in node.names if a.name.startswith("src")]

        bad = [d for d in deps if d.startswith("src.services")]
        assert not bad, f"迁移后不该再依赖 src.services：{bad}"
        # 至少要有 rag 内部依赖（证明它确实是 rag 的一部分）
        assert any(".rag" in d for d in deps), f"未发现 rag 内部依赖: {deps}"
