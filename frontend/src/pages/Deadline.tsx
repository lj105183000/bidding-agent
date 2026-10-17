/**
 * 截止日期监控页（P3）。
 *
 * ★ 这个页面此前被判定为「阻塞」——因为数据库里没有投标截止时间字段。
 *   现在它建立在**用户上传的招标文件**之上：`bid_deadline` 是从文档里抽出来的，
 *   落库后就能做"哪些项目快截止了"这类查询。
 *
 * ⚠️ 必须如实说明覆盖范围
 * ----------------------
 * 这不是"全网项目监控"，而是"**我上传过的项目**"监控。
 * 不说明这点，用户会以为系统漏报了项目。
 * 后端把说明放在 `coverage` 字段里，本页原样展示。
 *
 * 紧急度分四档（已过期单独一档）
 * ---------------------------
 * 「还有 1 天」和「已经截止」需要完全不同的行动，
 * 混在一起显示会误事，故 `expired` 独立于 `urgent`。
 */
import { useCallback, useEffect, useState } from 'react'
import {
  AlertTriangle,
  CalendarClock,
  CheckCircle2,
  Info,
  Loader2,
  RefreshCw,
  Trash2,
} from 'lucide-react'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Card from '../components/Card'
import EmptyState from '../components/EmptyState'
import PageHeader from '../components/PageHeader'
import Skeleton from '../components/Skeleton'
import { useToast } from '../components/Toast'
import {
  ApiError,
  deleteTenderDocument,
  fetchDeadlines,
  fetchTenderDocuments,
  type DeadlineResult,
  type TenderDocumentSummary,
  type Urgency,
} from '../lib/api'
import { formatMoney } from '../lib/format'

/** 紧急度 → 视觉与文案 */
const URGENCY_META: Record<Urgency, { color: 'danger' | 'warn' | 'brand' | 'ok' | 'neutral'; label: string }> = {
  expired: { color: 'neutral', label: '已截止' },
  urgent: { color: 'danger', label: '紧急（3 天内）' },
  soon: { color: 'warn', label: '临近（7 天内）' },
  normal: { color: 'brand', label: '正常' },
  unknown: { color: 'neutral', label: '未知' },
}

const WINDOW_OPTIONS = [
  { days: 7, label: '未来 7 天' },
  { days: 30, label: '未来 30 天' },
  { days: 90, label: '未来 90 天' },
  { days: 365, label: '未来一年' },
]

function daysText(d: number | null): string {
  if (d === null) return '—'
  if (d < 0) return `已过期 ${Math.abs(d).toFixed(0)} 天`
  if (d < 1) return `${(d * 24).toFixed(0)} 小时内`
  return `剩余 ${d.toFixed(0)} 天`
}

export default function Deadline() {
  const toast = useToast()
  const [days, setDays] = useState(30)
  const [data, setData] = useState<DeadlineResult | null>(null)
  const [allDocs, setAllDocs] = useState<TenderDocumentSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState<number | null>(null)

  const describe = (err: unknown) =>
    err instanceof ApiError
      ? err.status === 503
        ? '服务未就绪（请检查 PostgreSQL 是否启动）'
        : `请求失败：${err.message}`
      : '网络异常，请检查后端服务'

  const load = useCallback(async () => {
    setLoading(true)
    try {
      // 两个查询并行：截止监控 + 全部已解析文件（后者用于展示"没有截止时间"的那些）
      const [dl, docs] = await Promise.all([
        fetchDeadlines(days, true),
        fetchTenderDocuments(100),
      ])
      setData(dl)
      setAllDocs(docs.items)
    } catch (err) {
      toast.show(describe(err), 'error')
    } finally {
      setLoading(false)
    }
  }, [days, toast])

  useEffect(() => {
    void load()
  }, [load])

  const remove = async (id: number, name: string | null) => {
    setDeleting(id)
    try {
      await deleteTenderDocument(id)
      toast.show(`已删除「${name ?? id}」`, 'success')
      setAllDocs((prev) => prev.filter((d) => d.id !== id))
      void load()
    } catch (err) {
      toast.show(describe(err), 'error')
    } finally {
      setDeleting(null)
    }
  }

  const buckets = data?.summary.buckets ?? {}
  const noDeadline = allDocs.filter((d) => d.bid_deadline === null)

  return (
    <div>
      <PageHeader
        title="截止日期监控"
        description="已上传招标文件的投标截止时间与剩余天数"
        actions={
          <div className="flex items-center gap-2">
            <select
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="h-8 rounded-lg border border-slate-200 px-2 text-[12px] outline-none focus:border-brand-500"
            >
              {WINDOW_OPTIONS.map((o) => (
                <option key={o.days} value={o.days}>
                  {o.label}
                </option>
              ))}
            </select>
            <Button
              variant="secondary"
              size="sm"
              loading={loading}
              onClick={() => void load()}
              icon={<RefreshCw className="h-3.5 w-3.5" />}
            >
              刷新
            </Button>
          </div>
        }
      />

      {loading && !data && (
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-20 w-full" />
            ))}
          </div>
          <Skeleton className="h-48 w-full" />
        </div>
      )}

      {data && (
        <>
          {/* 概览分档 */}
          <div className="grid grid-cols-4 gap-4">
            {(
              [
                ['expired', '已截止', buckets.expired ?? 0, 'text-ink-400'],
                ['urgent', '3 天内', buckets.urgent ?? 0, 'text-danger'],
                ['soon', '7 天内', buckets.soon ?? 0, 'text-warn'],
                ['normal', `${days} 天内`, buckets.normal ?? 0, 'text-brand-700'],
              ] as const
            ).map(([key, label, count, cls]) => (
              <div key={key} className="rounded-xl border border-slate-200 bg-white p-4">
                <div className="text-[11px] text-ink-500">{label}</div>
                <div className={`mt-1 text-xl font-bold tabular-nums ${cls}`}>{count}</div>
                <div className="mt-0.5 text-[10px] text-ink-400">
                  {key === 'expired' ? '需要归档或放弃' : '个待投标项目'}
                </div>
              </div>
            ))}
          </div>

          {/* 覆盖范围说明：必须显眼——用户容易以为是全网监控 */}
          <div className="mt-4 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2.5">
            <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warn" />
            <div className="min-w-0 text-[11px] leading-5 text-ink-700">
              <b>覆盖范围：</b>
              <span dangerouslySetInnerHTML={{ __html: data.coverage.replace(/\*\*/g, '') }} />
            </div>
          </div>

          {/* 即将截止 */}
          <Card
            className="mt-4"
            title={`即将截止（${data.items.length} 项）`}
            subtitle={`按截止时间升序｜窗口 ${data.summary.window_days} 天`}
          >
            {data.items.length === 0 ? (
              <EmptyState
                icon={<CheckCircle2 className="h-6 w-6" />}
                title={`${days} 天内没有待投标项目`}
                description={
                  data.summary.with_deadline === 0
                    ? '还没有任何文件抽到截止时间。请先到「招标文件解析」页上传招标文件并保存'
                    : `已解析 ${data.summary.with_deadline} 份含截止时间的文件，但没有落在这个窗口内`
                }
              />
            ) : (
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="border-b border-slate-200 text-left text-[11px] text-ink-400">
                    <th className="w-24 py-2">紧急度</th>
                    <th className="py-2">项目名称</th>
                    <th className="w-28 py-2">采购人</th>
                    <th className="w-28 py-2 text-right">预算 / 限价</th>
                    <th className="w-40 py-2">投标截止</th>
                    <th className="w-28 py-2 text-right">剩余</th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((it) => {
                    const meta = URGENCY_META[it.urgency]
                    return (
                      <tr key={it.id} className="border-b border-slate-100 last:border-0">
                        <td className="py-1.5">
                          <Badge color={meta.color}>{meta.label}</Badge>
                        </td>
                        <td className="py-1.5">
                          <div className="max-w-[300px] truncate text-ink-900" title={it.project_name ?? ''}>
                            {it.project_name || it.filename}
                          </div>
                          {it.project_id && (
                            <div className="font-mono text-[10px] text-ink-400">{it.project_id}</div>
                          )}
                        </td>
                        <td className="py-1.5">
                          <div className="max-w-[140px] truncate text-[12px] text-ink-500" title={it.purchaser ?? ''}>
                            {it.purchaser || '—'}
                          </div>
                        </td>
                        <td className="py-1.5 text-right text-[12px] tabular-nums text-ink-600">
                          {it.budget ? formatMoney(it.budget) : '—'}
                          {it.max_price && it.max_price !== it.budget && (
                            <div className="text-[10px] text-ink-400">限价 {formatMoney(it.max_price)}</div>
                          )}
                        </td>
                        <td className="py-1.5 text-[12px] tabular-nums text-ink-700">
                          {it.bid_deadline?.slice(0, 16).replace('T', ' ') ?? '—'}
                        </td>
                        <td
                          className={`py-1.5 text-right text-[12px] font-medium tabular-nums ${
                            it.urgency === 'urgent' ? 'text-danger' : it.urgency === 'soon' ? 'text-warn' : 'text-ink-600'
                          }`}
                        >
                          {daysText(it.days_left)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            )}
          </Card>

          {/* 已截止 */}
          {data.expired.length > 0 && (
            <Card
              className="mt-4"
              title={`已截止（${data.expired.length} 项）`}
              subtitle="保留用于复盘：可查看当时的价格与资格要求"
            >
              <table className="w-full text-[13px]">
                <tbody>
                  {data.expired.map((it) => (
                    <tr key={it.id} className="border-b border-slate-100 last:border-0">
                      <td className="py-1.5 text-ink-500">
                        <div className="max-w-[320px] truncate" title={it.project_name ?? ''}>
                          {it.project_name || it.filename}
                        </div>
                      </td>
                      <td className="w-40 py-1.5 text-[12px] tabular-nums text-ink-400">
                        {it.bid_deadline?.slice(0, 16).replace('T', ' ') ?? '—'}
                      </td>
                      <td className="w-28 py-1.5 text-right text-[12px] tabular-nums text-ink-400">
                        {daysText(it.days_left)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* 未抽到截止时间的文件：如实列出，而不是隐藏 */}
          {noDeadline.length > 0 && (
            <Card
              className="mt-4"
              title={`未抽到截止时间（${noDeadline.length} 份）`}
              subtitle="这些文件不参与提醒——可能文件里没写，或抽取失败"
            >
              <div className="space-y-1">
                {noDeadline.map((d) => (
                  <div key={d.id} className="flex items-center gap-2 border-b border-slate-100 py-1.5 last:border-0">
                    <AlertTriangle className="h-3 w-3 shrink-0 text-warn" />
                    <span className="min-w-0 flex-1 truncate text-[12px] text-ink-600" title={d.filename}>
                      {d.project_name || d.filename}
                    </span>
                    <span className="shrink-0 text-[11px] text-ink-400">
                      完整度 {d.completeness !== null ? `${(d.completeness * 100).toFixed(0)}%` : '—'}
                    </span>
                    <button
                      onClick={() => void remove(d.id, d.project_name ?? d.filename)}
                      disabled={deleting === d.id}
                      className="shrink-0 rounded p-0.5 text-ink-300 hover:text-danger disabled:opacity-40"
                      title="删除这条记录"
                    >
                      {deleting === d.id ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <Trash2 className="h-3 w-3" />
                      )}
                    </button>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* 全部文件 */}
          {allDocs.length > 0 && (
            <Card className="mt-4" title={`全部已解析文件（${allDocs.length} 份）`}>
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="border-b border-slate-200 text-left text-[11px] text-ink-400">
                    <th className="w-10 py-2">id</th>
                    <th className="py-2">文件名 / 项目</th>
                    <th className="w-20 py-2">类型</th>
                    <th className="w-16 py-2 text-right">页数</th>
                    <th className="w-20 py-2 text-right">完整度</th>
                    <th className="w-32 py-2">解析时间</th>
                    <th className="w-10 py-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {allDocs.map((d) => (
                    <tr key={d.id} className="border-b border-slate-100 last:border-0">
                      <td className="py-1.5 font-mono text-[11px] text-ink-300">{d.id}</td>
                      <td className="py-1.5">
                        <div className="max-w-[320px] truncate text-ink-800" title={d.project_name ?? d.filename}>
                          {d.project_name || d.filename}
                        </div>
                        <div className="truncate text-[10px] text-ink-400" title={d.filename}>
                          {d.filename}
                        </div>
                      </td>
                      <td className="py-1.5">
                        <Badge color="neutral">{(d.kind ?? '?').toUpperCase()}</Badge>
                      </td>
                      <td className="py-1.5 text-right tabular-nums text-ink-500">{d.pages ?? '—'}</td>
                      <td className="py-1.5 text-right tabular-nums text-ink-500">
                        {d.completeness !== null ? `${(d.completeness * 100).toFixed(0)}%` : '—'}
                      </td>
                      <td className="py-1.5 text-[11px] tabular-nums text-ink-400">
                        {d.created_at?.slice(0, 16).replace('T', ' ') ?? '—'}
                      </td>
                      <td className="py-1.5">
                        <button
                          onClick={() => void remove(d.id, d.project_name ?? d.filename)}
                          disabled={deleting === d.id}
                          className="rounded p-0.5 text-ink-300 hover:text-danger disabled:opacity-40"
                          title="删除"
                        >
                          {deleting === d.id ? (
                            <Loader2 className="h-3 w-3 animate-spin" />
                          ) : (
                            <Trash2 className="h-3 w-3" />
                          )}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {allDocs.length === 0 && (
            <Card className="mt-4">
              <EmptyState
                icon={<CalendarClock className="h-6 w-6" />}
                title="还没有已解析的招标文件"
                description="到「招标文件解析」页上传 PDF/Word 并点「保存」，之后就能在这里看到截止提醒"
              />
            </Card>
          )}
        </>
      )}
    </div>
  )
}
