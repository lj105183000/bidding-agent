/**
 * 知识图谱关系可视化页。
 *
 * 数据来源：`/api/graph/overview` + `/api/graph/search` + `/api/graph/subgraph`
 *
 * 两个必须如实展示的局限（后端在 `legend.notes` 里给出，本页原样呈现）
 * ------------------------------------------------------------------
 * 1. **Transaction 节点不画入图**：实测 8730 个，只挂金额属性且无出边，
 *    画出来只是孤立点。中心节点的交易统计单独显示。
 * 2. **边不含金额**：图谱中没有「采购人 → 交易」的边，
 *    所以无法算出每条边对应的金额。想按供应商/采购人细分金额要用「集采分析」。
 *
 * 另外，节点数有上限（默认 40）。「空调」实测有 480 个邻居，
 * 全画出来既看不清也会让布局计算变慢，故按连接数截断并明确提示。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  Info,
  Layers,
  Loader2,
  Network,
  Search,
  Sliders,
  Sparkles,
} from 'lucide-react'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Card from '../components/Card'
import EmptyState from '../components/EmptyState'
import ForceGraph from '../components/ForceGraph'
import PageHeader from '../components/PageHeader'
import Skeleton from '../components/Skeleton'
import { useToast } from '../components/Toast'
import {
  ApiError,
  fetchGraphOverview,
  fetchSubgraph,
  searchGraphEntities,
  type GraphOverview,
  type SubgraphResponse,
} from '../lib/api'

const MAX_NODE_OPTIONS = [20, 30, 40, 60]

function money(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—'
  const n = Math.abs(v)
  if (n >= 100_000_000) return `¥${(v / 100_000_000).toFixed(2)}亿`
  if (n >= 10_000) return `¥${(v / 10_000).toFixed(1)}万`
  return `¥${v.toLocaleString('zh-CN', { maximumFractionDigits: 0 })}`
}

export default function GraphViz() {
  const toast = useToast()
  const debounceRef = useRef<number>()

  const [overview, setOverview] = useState<GraphOverview | null>(null)
  const [loadingOverview, setLoadingOverview] = useState(true)

  const [label, setLabel] = useState('SubjectMatter')
  const [keyword, setKeyword] = useState('空调')
  const [maxNodes, setMaxNodes] = useState(40)

  const [suggestions, setSuggestions] = useState<{ label: string; label_cn: string; name: string; degree: number }[]>([])
  const [showSuggest, setShowSuggest] = useState(false)

  const [graph, setGraph] = useState<SubgraphResponse | null>(null)
  const [loadingGraph, setLoadingGraph] = useState(false)

  useEffect(() => {
    fetchGraphOverview()
      .then(setOverview)
      .catch((err) => {
        toast.show(
          err instanceof ApiError ? `图谱信息拉取失败：${err.message}` : '无法连接图谱服务',
          'error',
        )
      })
      .finally(() => setLoadingOverview(false))
  }, [toast])

  // 关键词自动补全（带防抖，避免每个字符都打一次图谱）
  const onKeywordChange = (v: string) => {
    setKeyword(v)
    clearTimeout(debounceRef.current)
    if (v.trim().length < 1) {
      setSuggestions([])
      return
    }
    debounceRef.current = window.setTimeout(async () => {
      try {
        const items = await searchGraphEntities(v, 8)
        setSuggestions(items)
        setShowSuggest(true)
      } catch {
        setSuggestions([])
      }
    }, 300)
  }

  const load = useCallback(
    async (lbl = label, name = keyword, nodes = maxNodes) => {
      const clean = name.trim()
      if (!clean) {
        toast.show('请输入要探索的实体名称', 'warning')
        return
      }
      setLoadingGraph(true)
      setGraph(null)
      setShowSuggest(false)
      try {
        setGraph(await fetchSubgraph(lbl, clean, nodes))
      } catch (err) {
        const msg =
          err instanceof ApiError
            ? err.status === 400
              ? err.message
              : err.status === 503
                ? '知识图谱不可用（请检查 Neo4j 是否启动）'
                : `查询失败：${err.message}`
            : '网络异常，请检查后端服务'
        toast.show(msg, 'error')
      } finally {
        setLoadingGraph(false)
      }
    },
    [label, keyword, maxNodes, toast],
  )

  return (
    <div>
      <PageHeader
        title="图谱关系可视化"
        description="以任意实体为中心展开关系网络，直观查看标的物—供应商—采购单位的连接结构"
        actions={
          overview && (
            <div className="flex items-center gap-2">
              <Badge color="neutral">
                <Layers className="h-3 w-3" />
                {overview.total_nodes.toLocaleString('zh-CN')} 节点
              </Badge>
              <Badge color="neutral">
                <Network className="h-3 w-3" />
                {overview.total_relationships.toLocaleString('zh-CN')} 关系
              </Badge>
            </div>
          )
        }
      />

      {/* ── 查询条件 ─────────────────────────────────────────── */}
      <Card
        title="选择探索起点"
        subtitle="输入实体名称，或直接点下方的枢纽实体"
      >
        <div className="flex flex-wrap items-center gap-3">
          <select
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            className="h-9 rounded-lg border border-slate-200 px-2.5 text-[13px] outline-none focus:border-brand-500"
          >
            {(overview?.allowed_labels ?? [{ key: 'SubjectMatter', cn: '标的物' }]).map((l) => (
              <option key={l.key} value={l.key}>
                {l.cn}
              </option>
            ))}
          </select>

          <div className="relative min-w-[240px] flex-1">
            <input
              value={keyword}
              onChange={(e) => onKeywordChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void load()
                if (e.key === 'Escape') setShowSuggest(false)
              }}
              onFocus={() => suggestions.length > 0 && setShowSuggest(true)}
              placeholder="如：空调 / 合肥市 / 某某供应商"
              className="h-9 w-full rounded-lg border border-slate-200 px-3 text-[13px] outline-none focus:border-brand-500"
            />
            {showSuggest && suggestions.length > 0 && (
              <div className="absolute z-20 mt-1 w-full overflow-hidden rounded-lg border border-slate-200 bg-white shadow-lg">
                {suggestions.map((s) => (
                  <button
                    key={`${s.label}:${s.name}`}
                    onClick={() => {
                      setLabel(s.label)
                      setKeyword(s.name)
                      void load(s.label, s.name)
                    }}
                    className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[12px] hover:bg-brand-50"
                  >
                    <span className="shrink-0 rounded bg-ink-100 px-1.5 py-0.5 text-[10px] text-ink-500">
                      {s.label_cn}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-ink-900">{s.name}</span>
                    <span className="shrink-0 text-[10px] tabular-nums text-ink-400">
                      {s.degree}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>

          <select
            value={maxNodes}
            onChange={(e) => setMaxNodes(Number(e.target.value))}
            title="单次展示的最大节点数"
            className="h-9 rounded-lg border border-slate-200 px-2.5 text-[13px] outline-none focus:border-brand-500"
          >
            {MAX_NODE_OPTIONS.map((n) => (
              <option key={n} value={n}>
                最多 {n} 节点
              </option>
            ))}
          </select>

          <Button onClick={() => void load()} loading={loadingGraph} icon={<Search className="h-3.5 w-3.5" />}>
            展开关系
          </Button>
        </div>

        {/* 枢纽实体快捷入口 */}
        {loadingOverview && <Skeleton className="mt-3 h-16 w-full" />}
        {!loadingOverview && overview && overview.hub_entities.length > 0 && (
          <div className="mt-3">
            <div className="flex items-center gap-1.5 text-[11px] text-ink-400">
              <Sparkles className="h-3 w-3" />
              连接最多的实体（点一下直接展开）
            </div>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {overview.hub_entities.slice(0, 10).map((h) => (
                <button
                  key={`${h.label}:${h.name}`}
                  onClick={() => {
                    setLabel(h.label)
                    setKeyword(h.name)
                    void load(h.label, h.name)
                  }}
                  className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-2.5 py-1 text-[11px] text-ink-600 transition-colors hover:border-brand-300 hover:bg-brand-50"
                >
                  <span className="rounded bg-ink-100 px-1 py-0.5 text-[10px] text-ink-500">
                    {h.label_cn}
                  </span>
                  {h.name}
                  <span className="tabular-nums text-ink-300">{h.degree}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </Card>

      {/* ── 图区 ─────────────────────────────────────────────── */}
      {loadingGraph && (
        <Card className="mt-4">
          <div className="flex items-center gap-2 text-[12px] text-ink-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-brand-600" />
            正在抽取子图…
          </div>
          <Skeleton className="mt-3 h-[520px] w-full" />
        </Card>
      )}

      {!loadingGraph && !graph && (
        <Card className="mt-4">
          <EmptyState
            icon={<Network className="h-6 w-6" />}
            title="选择起点后展开关系网络"
            description="可以输入实体名称，也可以点上方「连接最多的实体」快速开始"
          />
        </Card>
      )}

      {!loadingGraph && graph && (
        <>
          {/* 中心实体信息 */}
          <div className="mt-4 grid grid-cols-4 gap-4">
            <div className="col-span-2 rounded-xl border border-slate-200 bg-white p-4">
              <div className="text-[11px] text-ink-400">中心实体</div>
              <div className="mt-1 flex items-center gap-2">
                <span className="rounded bg-ink-100 px-1.5 py-0.5 text-[10px] text-ink-500">
                  {graph.center.label_cn}
                </span>
                <span className="text-[15px] font-semibold text-ink-900">{graph.center.name}</span>
              </div>
              <div className="mt-1.5 text-[11px] text-ink-400">
                原图连接数 {graph.center.degree}
                {graph.center.txn_count !== undefined && `｜关联交易 ${graph.center.txn_count} 笔`}
              </div>
            </div>
            {graph.center.txn_count !== undefined && (
              <>
                <div className="rounded-xl border border-slate-200 bg-white p-4">
                  <div className="text-[11px] text-ink-400">交易金额中位数</div>
                  <div className="mt-1 text-lg font-bold tabular-nums text-ink-900">
                    {money(graph.center.amount_median)}
                  </div>
                  <div className="mt-0.5 text-[10px] text-ink-400">跨数量级，用中位数</div>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white p-4">
                  <div className="text-[11px] text-ink-400">交易金额合计</div>
                  <div className="mt-1 text-lg font-bold tabular-nums text-ink-900">
                    {money(graph.center.amount_sum)}
                  </div>
                  <div className="mt-0.5 text-[10px] text-ink-400">{graph.center.txn_count} 笔汇总</div>
                </div>
              </>
            )}
          </div>

          {/* 截断提示 */}
          {graph.truncated && (
            <div className="mt-4 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warn" />
              <div className="text-[11px] leading-5 text-ink-700">
                该实体的邻居共 <b>{graph.total_neighbors}</b> 个，图中按连接数展示了前{' '}
                <b>{graph.shown_nodes - 1}</b> 个。放大节点上限可看到更多，
                但节点过多会影响可读性与布局速度。
              </div>
            </div>
          )}

          <Card
            className="mt-4"
            title="关系网络"
            subtitle="节点大小＝连接数｜拖拽移动节点｜滚轮缩放｜空白处拖拽平移"
            extra={
              <span className="flex items-center gap-1 text-[11px] text-ink-400">
                <Sliders className="h-3 w-3" />
                {graph.edges.length} 条关系
              </span>
            }
          >
            <ForceGraph
              nodes={graph.nodes}
              edges={graph.edges}
              onNodeClick={(n) => {
                // 点邻居 → 把它当作新的中心再展开一层（探索式浏览）
                if (!n.is_center) {
                  setLabel(n.label)
                  setKeyword(n.name)
                  void load(n.label, n.name)
                }
              }}
            />
          </Card>

          {/* 后端给出的局限说明，原样展示 */}
          {graph.legend.notes.length > 0 && (
            <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
              <div className="flex items-center gap-1.5 text-[12px] font-semibold text-ink-600">
                <Info className="h-3.5 w-3.5" />
                图的结构说明
              </div>
              <ul className="mt-2 space-y-1">
                {graph.legend.notes.map((n, i) => (
                  <li key={i} className="flex gap-1.5 text-[11px] leading-5 text-ink-500">
                    <span className="text-ink-300">•</span>
                    <span>{n}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* 邻居列表（图看不清时的补充） */}
          {graph.nodes.length > 1 && (
            <Card className="mt-4" title="邻居明细" subtitle="按连接数降序，点名称可继续展开">
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="border-b border-slate-200 text-left text-[11px] text-ink-400">
                    <th className="w-10 py-2">#</th>
                    <th className="w-20 py-2">类型</th>
                    <th className="py-2">名称</th>
                    <th className="w-24 py-2">关系</th>
                    <th className="w-20 py-2 text-right">连接数</th>
                  </tr>
                </thead>
                <tbody>
                  {graph.nodes.slice(1).map((n, i) => {
                    const edge = graph.edges.find((e) => e.target === n.id)
                    return (
                      <tr key={n.id} className="border-b border-slate-100 last:border-0">
                        <td className="py-1.5 text-[11px] text-ink-300">{i + 1}</td>
                        <td className="py-1.5">
                          <span className="rounded bg-ink-100 px-1.5 py-0.5 text-[10px] text-ink-500">
                            {n.label_cn}
                          </span>
                        </td>
                        <td className="py-1.5">
                          <button
                            onClick={() => {
                              setLabel(n.label)
                              setKeyword(n.name)
                              void load(n.label, n.name)
                            }}
                            className="text-left text-brand-700 hover:underline"
                          >
                            {n.name}
                          </button>
                        </td>
                        <td className="py-1.5 text-[11px] text-ink-500">{edge?.rel_cn ?? '—'}</td>
                        <td className="py-1.5 text-right tabular-nums text-ink-600">{n.degree}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
