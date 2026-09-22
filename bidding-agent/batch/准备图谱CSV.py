"""从原始抽取结果 xlsx 派生两份 CSV，供 Neo4j 的 LOAD CSV 建知识图谱。

用途
----
把 `data/processed/` 下的原始表转成图谱能直接吃进去的两张表：

    data/processed/标的物_交易频次.csv                        标的物 -> 出现次数（先建节点、带热度）
    data/processed/招标采购标的物信息提取训练数据_2_知识图谱.csv   一次交易一行（后建关系）

为什么单独放在 batch/ 而不是塞进 src/
------------------------------------
这是**离线的一次性数据准备**（跑完就完事，不进服务进程），依赖只有 pandas。
放进 src/ 会让"运行期代码"和"数据准备脚本"混在一起，也会让 CI 的 import 链多背一份
pandas 依赖。

为什么写 utf-8 而不是 utf-8-sig
-------------------------------
给 Neo4j 的 `LOAD CSV WITH HEADERS` 用。带 BOM 时第一个列名会变成 `"\ufeff标的物"`，
于是 `row.标的物` 取不到值——而报错信息完全看不出是 BOM 干的，很难从症状倒推回病因。
同理，字段内的换行也要先压掉（见 `clean_text`）。

运行
----
    python batch/准备图谱CSV.py                 # 用默认输入
    python batch/准备图谱CSV.py 别的文件.xlsx    # 换一份输入（可选，位置参数）

输入文件不会被修改，只读。
"""

import sys
from pathlib import Path

import pandas as pd

# 以脚本所在位置反推仓库根（batch/ 的上一级），这样从任何目录执行都指向同一份数据；
# 若改成相对当前目录（Path("data/processed")），在别处执行就会读到别的目录去。
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_INPUT = PROCESSED_DIR / "招标采购标的物信息提取训练数据_2_标的物.xlsx"

FREQ_CSV = PROCESSED_DIR / "标的物_交易频次.csv"
GRAPH_CSV = PROCESSED_DIR / "招标采购标的物信息提取训练数据_2_知识图谱.csv"

# 源列名 -> 输出列名。源表里**没有**叫「标的物」的列，只有「核心标的物」，
# 输出侧才统一叫「标的物」（图谱里的节点名）。
SRC_SUBJECT = "核心标的物"
SUBJECT_FALLBACK = "标的物"  # 换过来源的表可能就叫这个，缺列时兜底，但会打印提示
OUT_SUBJECT = "标的物"
OUT_PURCHASER = "采购商"
OUT_AGENCY = "代理机构"
OUT_SUPPLIER = "供应商"
OUT_AMOUNT = "交易额"
OUT_LOCATION = "所在地"

# 「知识图谱.csv」要用到的源列（标的物单列处理，见 pick_subject_column）
GRAPH_SOURCE_COLUMNS = {
    "采购商": "采购人",
    "代理机构": "代理机构",
    "供应商": "中标人",
    "交易额": "中标金额",
    "所在地": "市区",
}

# 这些写法的意思是"没有值"，先归一成空再统计，免得把它们算进"解析失败"的告警里。
NULL_LIKE = {"", "-", "—", "--", "无", "None", "none", "nan", "NaN", "NULL", "null"}

PREVIEW_ROWS = 3


def read_table(path: Path) -> tuple[pd.DataFrame, str]:
    """读 xlsx，返回 (数据, 用到的 sheet 名)。

    多 sheet 时挑**行数最多**的那张并打印全部 sheet 的行数：抽取结果常把主表放在
    非第一张 sheet 上（第一张可能是说明或样本），默认只读第一张会静默拿到错数据——
    在本脚本里表现为"行数少得离谱"，而打印出来就一眼可见。
    """
    sheets = pd.read_excel(path, sheet_name=None)
    if len(sheets) == 1:
        name = next(iter(sheets))
    else:
        name = max(sheets, key=lambda key: len(sheets[key]))
        sizes = "、".join(f"{key}({len(value)} 行)" for key, value in sheets.items())
        print(f"该文件有 {len(sheets)} 张工作表：{sizes} -> 取行数最多的「{name}」")

    frame = sheets[name].copy()
    # 列名去掉空白与 BOM：Excel 里手改过表头会留下尾随空格或不可见字符，
    # 而 `df["核心标的物"]` 会以 KeyError 形式报"没有这一列"，看着跟没这一列一模一样。
    # BOM 用转义写（"\ufeff"）而不是敲一个不可见字符：后者在编辑器里存一次就可能被吃掉，
    # 而症状是"这行突然不生效了"，几乎不可能查出来
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    return frame, name


def pick_subject_column(frame: pd.DataFrame) -> str:
    """确定用哪一列当「标的物」；两列都没有就抛 KeyError（由 main 打印成中文提示）。

    缺列时不猜也不静默跳过：图谱里所有关系都挂在标的物节点上，拿错列 = 整张图谱白建。
    """
    if SRC_SUBJECT in frame.columns:
        return SRC_SUBJECT
    if SUBJECT_FALLBACK in frame.columns:
        print(f"提示：没有「{SRC_SUBJECT}」列，改用「{SUBJECT_FALLBACK}」列——"
              f"确认这是同一份数据，别把两种来源的图谱混在一起")
        return SUBJECT_FALLBACK
    raise KeyError(f"输入表里既没有「{SRC_SUBJECT}」也没有「{SUBJECT_FALLBACK}」列，无法建图谱")


def clean_text(series: pd.Series) -> pd.Series:
    """文本列归一：压掉换行与连续空白、去首尾空白、空值与占位符统一成空字符串。

    空值统一成 `""`（而不是留 NaN）是给 Neo4j 用：`LOAD CSV` 把空字段读成 null，
    而这里"代理机构为空"是**正常的**业务事实（大量项目没有代理机构），
    空字符串进图谱后是"有节点但名为空"，与"没有这个节点"能区分开，也便于事后筛。
    压空白是防 CSV 结构被破坏：字段里的换行虽然会被引号包住，但 Neo4j 的 LOAD CSV
    对多行字段很挑剔，且这类脏数据（"合肥市\n"）肉眼几乎看不出来。

    `-` / `无` / `NULL` 这类**占位符**也要归一成空：它们写的是"这里没有值"，
    但文本列不做这步就会被当成正常值。本表「核心标的物」真有 4 行写「无」——
    不处理的话图谱里会凭空多出一个名叫「无」的标的物节点，还会进频次排行。
    """
    text = series.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    # mask 先置空再 fillna：一次盖住"原生 NaN"和"占位符"两种空。
    # 注意判定必须在 strip 之后，否则 " - " 这种带空白的占位符会漏掉。
    return text.mask(text.isin(NULL_LIKE)).fillna("")


def to_amount(series: pd.Series) -> pd.Series:
    """中标金额 → 数值列；空值/解析不了的落成 NaN（写进 CSV 就是空字段）。

    先去掉千分位与空格再转数字：Excel 里这类金额列常是"数字与文本混排"，
    `"1,684,000"` 直接 `to_numeric` 会变成 NaN——金额被静默吞掉比报错严重得多，
    所以解析失败的行数（连同几个原值样例）会打印出来，让人知道丢了几条。
    不四舍五入：float64 在千万量级（本表最大约 8.8e7）有效位富余，
    金额按原值出入，尾随的 ".0" 是浮点的正常写法，Neo4j 的 toFloat() 照收。
    """
    text = series.astype("string").str.strip()
    for separator in (",", "，", " "):
        text = text.str.replace(separator, "", regex=False)
    text = text.mask(text.isin(NULL_LIKE))

    numeric = pd.to_numeric(text, errors="coerce")
    failed = text.notna() & numeric.isna()
    if failed.any():
        examples = "、".join(map(str, text[failed].unique()[:5]))
        print(f"！有 {int(failed.sum())} 行「{OUT_AMOUNT}」不是数字，按空值处理（原值示例：{examples}）")
    return numeric


def build_frequency(frame: pd.DataFrame, subject_column: str) -> pd.DataFrame:
    """标的物 -> 出现次数，按频次降序。"""
    subjects = clean_text(frame[subject_column])
    counts = subjects[subjects != ""].value_counts()
    freq = counts.rename_axis(OUT_SUBJECT).reset_index(name="交易频次")
    # 次级排序键：频次相同的标的物之间必须有稳定顺序，否则同一份输入跑两次得到的 CSV
    # 行序不同，diff 全是噪音，也没法判断"这次的输出和上次是否真的一样"。
    return freq.sort_values(["交易频次", OUT_SUBJECT], ascending=[False, True]).reset_index(drop=True)


def build_graph(frame: pd.DataFrame, subject_column: str) -> pd.DataFrame:
    """一次交易一行：标的物 / 采购商 / 代理机构 / 供应商 / 交易额 / 所在地。

    丢"标的物为空"的整行，是因为图谱里所有关系都从标的物节点出发：
    没有标的物的记录在图上没有挂载点，硬塞进去只能连到一个空名字的节点上，
    既污染节点集合，查的时候也永远查不到它。其余列的缺失都是可接受的（留空字符串）。

    只挑这 6 列输出（而不是把原始 21 列全带上）：CSV 是给 LOAD CSV 用的，
    多出来的列既不会被用到，还会让"这次导入到底用了哪些字段"变得不清楚。
    """
    missing = [source for source in GRAPH_SOURCE_COLUMNS.values() if source not in frame.columns]
    if missing:
        raise KeyError(f"输入表缺少这些列：{'、'.join(missing)}")

    graph = pd.DataFrame(
        {
            OUT_SUBJECT: clean_text(frame[subject_column]),
            OUT_PURCHASER: clean_text(frame[GRAPH_SOURCE_COLUMNS["采购商"]]),
            OUT_AGENCY: clean_text(frame[GRAPH_SOURCE_COLUMNS["代理机构"]]),
            OUT_SUPPLIER: clean_text(frame[GRAPH_SOURCE_COLUMNS["供应商"]]),
            OUT_AMOUNT: to_amount(frame[GRAPH_SOURCE_COLUMNS["交易额"]]),
            OUT_LOCATION: clean_text(frame[GRAPH_SOURCE_COLUMNS["所在地"]]),
        }
    )
    before = len(graph)
    graph = graph[graph[OUT_SUBJECT] != ""].reset_index(drop=True)
    dropped = before - len(graph)
    if dropped:
        print(f"丢掉了 {dropped} 行没有标的物的记录（在图上没有挂载点）")
    return graph


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """写 CSV 并打印前几行 + 总行数，供肉眼核对。

    `lineterminator="\\n"` 固定成 LF：pandas 默认跟随操作系统（Windows 上写 \\r\\n），
    同一份数据在两个平台上产出的文件字节不同，浪费 diff；Neo4j 两种都收。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")

    print(f"\n== {path.name} ==")
    print(f"前 {PREVIEW_ROWS} 行：")
    print(frame.head(PREVIEW_ROWS).to_string(index=False) if len(frame) else "（空表）")
    print(f"总行数：{len(frame)}")
    print(f"已写出：{path}")


def main(argv: list[str]) -> int:
    source = Path(argv[1]) if len(argv) > 1 else DEFAULT_INPUT
    if not source.exists():
        print(f"找不到输入文件：{source}")
        print("确认路径，或用位置参数指定：python batch/准备图谱CSV.py 你的文件.xlsx")
        return 1

    print(f"输入：{source}")
    frame, sheet = read_table(source)
    print(f"工作表「{sheet}」：{len(frame)} 行 × {len(frame.columns)} 列")
    print(f"列名：{'、'.join(map(str, frame.columns))}")

    try:
        subject_column = pick_subject_column(frame)
        graph = build_graph(frame, subject_column)
    except KeyError as error:
        print(f"！{error.args[0]}")
        return 1

    write_csv(build_frequency(frame, subject_column), FREQ_CSV)
    write_csv(graph, GRAPH_CSV)
    return 0


if __name__ == "__main__":
    # 中文列名在终端里按东亚宽度对齐，否则表头与数据列会错位，肉眼检查反而更费劲
    pd.set_option("display.unicode.east_asian_width", True)
    sys.exit(main(sys.argv))
