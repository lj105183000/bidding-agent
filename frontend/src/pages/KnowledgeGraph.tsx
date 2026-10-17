/**
 * 知识图谱查询页。
 *
 * 与「智能问答」页的分工：
 *   - 问答页：用户用自然语言提问，Agent 自己决定要不要查图谱、查哪一类
 *   - 本页  ：用户在界面上**显式指定**查询类型与参数，直接看原始数据
 *
 * 为什么不让问答页复用：Agent 返回的是给 LLM 看的 markdown 文本（字段被本地化、
 * 条数被截断），前端解析它既脆弱又丢信息。本页走后端 `/api/graph/query`，
 * 拿到的是结构化 JSON，能画真表格、标徽章、显示完整金额。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import {
  AlertCircle,
  Boxes,
  Database,
  MessageSquareQuote,
  Network,
  Play,
  RefreshCw,
  Sparkles,
} from 'lucide-react'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Card from '../components/Card'
import EmptyState from '../components/EmptyState'
import PageHeader from '../components/PageHeader'
import Skeleton from '../components/Skeleton'
import { useToast } from '../components/Toast'
import { ApiError, fetchGraphTypes, queryGraph, type GraphPayload } from '../lib/api'
import {
  DETAIL_SECTIONS,
  LABEL_NAMES,
  QUERY_TYPES,
  RELATION_NAMES,
  getQueryTypeMeta,
  type ColumnDef,
  type GraphNavPayload,
} from '../data/graph'

const LIMIT_OPTIONS = [10, 20, 50, 100]

type Status = 'idle' | 'loading' | 'done' | 'error'

/** 把后端状态码翻译成用户能照做的提示 */
function describeError(err: unknown): { title: string; hint: string } {
  if (err instanceof ApiError) {
    if (err.status === 503) {
      return {
        title: '知识图谱暂时不可用',
        hint: 'Neo4j 实例可能未启动。若使用本地 Docker：确认 Docker Desktop 正在运行，且 bid-neo4j 容器状态为 Up。',
      }
    }
    if (err.status === 429) {
      return { title: '请求过于频繁', hint: '请稍候几秒再试（限流 30 次 / 分钟）。' }
    }
    if (err.status === 400) {
      return { title: '查询参数不合法', hint: err.message }
    }
    return { title: `请求失败（HTTP ${err.status}）`, hint: err.message }
  }
  return { title: '网络错误', hint: '无法连接后端 :8001，确认后端服务已启动。' }
}

/** 单元格渲染：按列定义的类型走不同样式，未知字段一律安全转字符串 */
function Cell({ column, row }: { column: ColumnDef; row: Record<string, unknown> }) {
  const raw = row[column.key]

  if (raw === null || raw === undefined || raw === '') {
    return <span className="text-ink-300">—</span>
  }

  if (column.type === 'number') {
    const num = typeof raw === 'number' ? raw : Number(raw)
    if (Number.isFinite(num)) {
      return <span className="tabular-nums text-ink-900">{num.toLocaleString('zh-CN')}</span>
    }
    return <span className="text-ink-700">{String(raw)}</span>
  }

  if (column.type === 'badge') {
    const text = String(raw)
    // label 列返回英文标签（SubjectMatter），relation 列返回英文关系类型
    const pretty = LABEL_NAMES[text] ?? RELATION_NAMES[text] ?? text
    const color = column.key === 'label' ? 'brand' : 'neutral'
    return <Badge color={color}>{pretty}</Badge>
  }

  const text = String(raw)
  return (
    <span className={column.type === 'mono' ? 'font-mono text-[11px]' : ''} title={text}>
      {text}
    </span>
  )
}

/** 结果条数/形状归一：dict 与 list 都转成"能判断是否为空"的形式 */
function payloadSize(payload: GraphPayload): number {
  if (Array.isArray(payload)) return payload.length
  if (payload && typeof payload === 'object') return Object.keys(payload).length
  return 0
}

export default function KnowledgeGraph() {
  const toast = useToast()
  const location = useLocation()

  /** 首页示例跳转带来的载荷（见 data/graph.ts 的 buildExamplePayload） */
  const navPayload = (location.state ?? null) as GraphNavPayload | null

  const [queryType, setQueryType] = useState('top_traded')
  const [entity, setEntity] = useState('')
  const [limit, setLimit] = useState(20)

  const [status, setStatus] = useState<Status>('idle')
  const [payload, setPayload] = useState<GraphPayload | null>(null)
  const [count, setCount] = useState(0)
  const [errorInfo, setErrorInfo] = useState<{ title: string; hint: string } | null>(null)
  /** 回显"从首页点进来的那个问题"，让用户确认跑的是自己点的那条 */
  const [askedQuestion, setAskedQuestion] = useState('')

  const [typesReady, setTypesReady] = useState(false)
  const [graphReady, setGraphReady] = useState<boolean | null>(null)

  const meta = useMemo(() => getQueryTypeMeta(queryType), [queryType])

  // 从后端拉可用类型：只用来判断图谱健康状态与校验清单是否一致，
  // 展示用的中文名/图标/示例仍取本地配置（后端不提供这些展示信息）
  const loadTypes = useCallback(() => {
    fetchGraphTypes()
      .then((res) => {
        setGraphReady(res.graph_ready)
        setTypesReady(true)
        const known = new Set(res.types.map((t) => t.query_type))
        const missing = QUERY_TYPES.filter((t) => !known.has(t.query_type))
        if (missing.length > 0) {
          // 后端有而前端没登记的展示元数据：不是错误，但值得提醒开发者
          console.warn('[knowledge-graph] 后端返回了未登记的查询类型：', missing.map((m) => m.query_type))
        }
      })
      .catch((err) => {
        setTypesReady(true)
        setGraphReady(false)
        const info = describeError(err)
        setErrorInfo(info)
        setStatus('error')
      })
  }, [])

  useEffect(() => {
    loadTypes()
  }, [loadTypes])

  const run = useCallback(
    async (overrideType?: string, overrideEntity?: string) => {
      const qt = overrideType ?? queryType
      const target = getQueryTypeMeta(qt)
      const entityValue = (overrideEntity ?? entity).trim()

      if (target?.requiresEntity && !entityValue) {
        toast.show(`「${target.label}」需要先填写实体名称`, 'warning')
        return
      }

      setStatus('loading')
      setErrorInfo(null)
      setPayload(null)

      try {
        const res = await queryGraph({
          query_type: qt,
          entity: entityValue,
          limit,
        })
        setPayload(res.data)
        setCount(res.count)
        setStatus('done')
        if (payloadSize(res.data) === 0) {
          toast.show('查询成功，但没有匹配的数据', 'info')
        }
      } catch (err) {
        const info = describeError(err)
        setErrorInfo(info)
        setStatus('error')
        toast.show(info.title, 'error')
      }
    },
    [queryType, entity, limit, toast],
  )

  /**
   * 首页示例跳转：预填表单并自动执行一次。
   *
   * `useRef` 守卫不能用 `useEffect` 的依赖数组代替——React 18 的 StrictMode 在开发环境
   * 会把 effect 跑两次，那样同一条示例会打两次 Neo4j。用 ref 记住"处理过哪个 state 对象"，
   * 同一个对象（含 StrictMode 的第二次执行）只处理一次；
   * 返回本页再点别的问题会带来**新的** state 对象，因此仍会正常触发。
   */
  const consumedState = useRef<GraphNavPayload | null>(null)

  useEffect(() => {
    if (!navPayload || !navPayload.autoRun) return
    if (consumedState.current === navPayload) return
    consumedState.current = navPayload

    const nextType = navPayload.query_type ?? 'top_traded'
    const nextEntity = navPayload.entity ?? ''
    setQueryType(nextType)
    setEntity(nextEntity)
    setAskedQuestion(navPayload.question ?? '')

    const target = getQueryTypeMeta(nextType)
    if (target?.requiresEntity && !nextEntity) {
      // 示例没带实体（配置漏写）：只预填类型，提示用户补填，不空跑一次必然失败的查询
      toast.show('该示例需要实体名称，请补充后点「执行查询」', 'warning')
      return
    }
    void run(nextType, nextEntity)
  }, [navPayload, run, toast])

  /** 点类型卡片：不依赖实体的类型直接开跑，需要实体的把焦点交给输入框 */
  const pickType = (nextType: string) => {
    const target = getQueryTypeMeta(nextType)
    setQueryType(nextType)
    setErrorInfo(null)
    if (!target?.requiresEntity) {
      setEntity('')
      void run(nextType, '')
    } else {
      setPayload(null)
      setStatus('idle')
    }
  }

  const showResult = status === 'done' && payload !== null && payloadSize(payload) > 0

  return (
    <div>
      <PageHeader
        title="知识图谱"
        description="基于 Neo4j 的招投标实体关系直查 · 标的物 / 采购单位 / 供应商 / 代理机构 / 地区"
        actions={
          <>
            <Badge color={graphReady === null ? 'neutral' : graphReady ? 'ok' : 'danger'}>
              <Database className="h-3 w-3" />
              {graphReady === null ? '检测中…' : graphReady ? 'Neo4j 已连接' : 'Neo4j 未连接'}
            </Badge>
            <Button
              variant="secondary"
              size="sm"
              icon={<RefreshCw className="h-3.5 w-3.5" />}
              onClick={() => {
                loadTypes()
                toast.show('已重新检测图谱状态', 'info')
              }}
            >
              重新检测
            </Button>
          </>
        }
      />

      {/* ── 从首页示例跳转时的回显 ─────────────────────────────────── */}
      {askedQuestion && (
        <div className="mt-4 flex items-start gap-2 rounded-lg border border-brand-100 bg-brand-50 px-4 py-3">
          <MessageSquareQuote className="mt-0.5 h-3.5 w-3.5 shrink-0 text-brand-600" />
          <div className="min-w-0">
            <div className="text-[11px] text-ink-400">你从首页点了这个问题</div>
            <div className="mt-0.5 text-[13px] font-medium text-ink-900">{askedQuestion}</div>
          </div>
        </div>
      )}

      {/* ── 查询类型选择 ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
        {QUERY_TYPES.map((t) => {
          const active = t.query_type === queryType
          return (
            <button
              key={t.query_type}
              onClick={() => pickType(t.query_type)}
              className={`rounded-xl border p-3 text-left transition-all ${
                active
                  ? 'border-brand-500 bg-brand-50 shadow-sm'
                  : 'border-slate-200 bg-white hover:border-brand-200 hover:bg-brand-50/40'
              }`}
            >
              <div className="flex items-center gap-2">
                <t.icon className={`h-4 w-4 shrink-0 ${active ? 'text-brand-600' : 'text-ink-400'}`} />
                <span
                  className={`text-[13px] font-semibold ${active ? 'text-brand-700' : 'text-ink-900'}`}
                >
                  {t.label}
                </span>
              </div>
              <div className="mt-1.5 text-[11px] leading-4 text-ink-400">{t.description}</div>
              <div className="mt-2 font-mono text-[10px] text-ink-300">{t.query_type}</div>
            </button>
          )
        })}
      </div>

      {/* ── 查询参数 ────────────────────────────────────────────────── */}
      <Card
        className="mt-6"
        title={`查询参数 · ${meta?.label ?? queryType}`}
        subtitle={meta?.description}
        extra={
          <select
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="h-8 rounded-lg border border-slate-200 bg-white px-2 text-xs text-ink-700 outline-none focus:border-brand-500"
          >
            {LIMIT_OPTIONS.map((n) => (
              <option key={n} value={n}>
                条数上限 {n}
              </option>
            ))}
          </select>
        }
      >
        <div className="flex items-start gap-3">
          <div className="min-w-0 flex-1">
            <label className="text-[11px] text-ink-400">
              实体名称 {meta?.requiresEntity ? <span className="text-danger">*</span> : '（该类型无需填写）'}
            </label>
            <input
              value={entity}
              onChange={(e) => setEntity(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void run()
              }}
              disabled={!meta?.requiresEntity}
              placeholder={meta?.requiresEntity ? (meta?.entityPlaceholder ?? '输入实体名称') : '—'}
              className="mt-1 h-9 w-full rounded-lg border border-slate-200 px-3 text-[13px] text-ink-900 outline-none transition-colors focus:border-brand-500 disabled:bg-ink-50 disabled:text-ink-400"
            />
          </div>
          <div className="shrink-0 pt-[18px]">
            <Button
              onClick={() => void run()}
              loading={status === 'loading'}
              icon={<Play className="h-3.5 w-3.5" />}
            >
              执行查询
            </Button>
          </div>
        </div>

        {/* 示例问题：点击即填并执行，降低上手成本 */}
        {meta && meta.examples.length > 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span className="text-[11px] text-ink-400">示例：</span>
            {meta.examples.map((ex) => (
              <button
                key={ex}
                onClick={() => {
                  // 示例是"自然语言问题"，摘出其中的实体名并不现实——
                  // 有实体的类型直接填第一个示例的实体部分（去掉问号与疑问词）
                  if (meta.requiresEntity) {
                    const guess = ex.replace(/[？?]/g, '').replace(/(有哪些?|是什么|是什么？)/g, '')
                    setEntity(guess)
                    void run(queryType, guess)
                  } else {
                    void run(queryType, '')
                  }
                }}
                className="rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] text-ink-500 transition-colors hover:border-brand-200 hover:bg-brand-50 hover:text-brand-700"
              >
                {ex}
              </button>
            ))}
          </div>
        )}
      </Card>

      {/* ── 结果区 ──────────────────────────────────────────────────── */}
      <Card
        className="mt-6"
        title={meta?.resultTitle ?? '查询结果'}
        subtitle={
          showResult
            ? `共 ${count} 条${Array.isArray(payload) && count >= limit ? `（已达上限 ${limit}，可调高条数上限）` : ''}`
            : undefined
        }
        extra={
          showResult ? (
            <Badge color="neutral">
              <Network className="h-3 w-3" />
              {queryType}
            </Badge>
          ) : undefined
        }
      >
        {status === 'loading' && (
          <div className="space-y-3">
            <Skeleton className="h-6 w-1/3" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-4/5" />
          </div>
        )}

        {status === 'error' && errorInfo && (
          <div className="rounded-lg border-l-2 border-danger bg-red-50 px-4 py-3">
            <div className="flex items-center gap-2 text-[13px] font-semibold text-danger">
              <AlertCircle className="h-4 w-4" />
              {errorInfo.title}
            </div>
            <div className="mt-1 text-xs leading-5 text-ink-600">{errorInfo.hint}</div>
          </div>
        )}

        {status !== 'loading' && status !== 'error' && !showResult && (
          <EmptyState
            icon={<Boxes className="h-6 w-6" />}
            title={status === 'done' ? '没有匹配的数据' : '选择查询类型开始'}
            description={
              status === 'done'
                ? '试试换个实体名，或改用「实体检索」做模糊匹配'
                : '点击上方任一类型卡片；需要实体的类型请填写名称后点「执行查询」'
            }
          />
        )}

        {showResult && payload && (
          <div>
            {/* graph_stats：对象 → 统计卡片 */}
            {queryType === 'graph_stats' && !Array.isArray(payload) && (
              <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
                {Object.entries(payload).map(([key, value]) => (
                  <div key={key} className="rounded-xl border border-slate-200 p-4">
                    <div className="text-[11px] text-ink-400">{LABEL_NAMES[key] ?? key}</div>
                    <div className="mt-1 text-[22px] font-bold leading-7 text-ink-900 tabular-nums">
                      {Number(value).toLocaleString('zh-CN')}
                    </div>
                    <div className="mt-1 font-mono text-[10px] text-ink-300">{key}</div>
                  </div>
                ))}
              </div>
            )}

            {/* entity_detail：对象 → 分段列表 */}
            {queryType === 'entity_detail' && !Array.isArray(payload) && (
              <div className="space-y-5">
                {typeof payload.subject === 'string' && (
                  <div className="rounded-lg border border-brand-100 bg-brand-50 px-4 py-3">
                    <div className="text-[11px] text-ink-400">标的物</div>
                    <div className="mt-0.5 text-[15px] font-semibold text-brand-700">
                      {payload.subject}
                    </div>
                  </div>
                )}
                <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                  {DETAIL_SECTIONS.map(({ key, title }) => {
                    const values = payload[key]
                    if (!Array.isArray(values) || values.length === 0) return null
                    return (
                      <div key={key} className="rounded-xl border border-slate-200 p-4">
                        <div className="flex items-center justify-between">
                          <span className="text-[13px] font-semibold text-ink-900">{title}</span>
                          <Badge color="neutral">{values.length} 项</Badge>
                        </div>
                        <ul className="mt-3 space-y-1.5">
                          {values.slice(0, 50).map((item, i) => (
                            <li key={`${key}-${i}`} className="text-xs leading-5 text-ink-700">
                              {typeof item === 'object' && item !== null ? (
                                // 交易记录这类子对象：键值对拼一行
                                <span>
                                  {Object.entries(item as Record<string, unknown>)
                                    .filter(([, v]) => v !== null && v !== undefined && v !== '')
                                    .map(([k, v]) => `${k}：${v}`)
                                    .join('；')}
                                </span>
                              ) : (
                                <span>{String(item)}</span>
                              )}
                            </li>
                          ))}
                        </ul>
                        {values.length > 50 && (
                          <div className="mt-2 text-[11px] text-ink-400">
                            仅显示前 50 项，共 {values.length} 项
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* 其余 4 类：数组 → 表格 */}
            {Array.isArray(payload) && (
              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-left text-[13px]">
                  <thead>
                    <tr className="border-b border-slate-200">
                      <th className="w-12 px-2 py-2 text-[11px] font-medium text-ink-400">#</th>
                      {(meta?.columns ?? []).map((col) => (
                        <th
                          key={col.key}
                          className={`px-2 py-2 text-[11px] font-medium text-ink-400 ${
                            col.type === 'number' ? 'text-right' : ''
                          }`}
                        >
                          {col.label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {payload.map((row, i) => (
                      <tr
                        key={i}
                        className="border-b border-slate-100 transition-colors hover:bg-ink-50/60"
                      >
                        <td className="px-2 py-2 text-[11px] text-ink-300 tabular-nums">{i + 1}</td>
                        {(meta?.columns ?? []).map((col) => (
                          <td
                            key={col.key}
                            className={`max-w-[380px] truncate px-2 py-2 text-ink-700 ${
                              col.type === 'number' ? 'text-right' : ''
                            }`}
                          >
                            <Cell column={col} row={row} />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* ── 说明 ────────────────────────────────────────────────────── */}
      {typesReady && (
        <div className="mt-4 flex items-start gap-2 text-[11px] leading-5 text-ink-400">
          <Sparkles className="mt-0.5 h-3 w-3 shrink-0" />
          <span>
            本页直连 Neo4j 读取原始数据，不经过大模型，因此结果与「智能问答」页可能有细微差异
            （问答页会由模型组织语言并截断长列表）。图谱数据来自
            <span className="font-mono"> data/processed/</span> 下的两份 CSV，
            重建命令：<span className="font-mono">python main.py graph --clear</span>
          </span>
        </div>
      )}
    </div>
  )
}
