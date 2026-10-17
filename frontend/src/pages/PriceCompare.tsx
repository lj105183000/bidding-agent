/**
 * 智能商品比价页。
 *
 * 数据来源：`POST /api/price-compare`（后端编排：视觉提取商品 → 并行检索各平台 → 择优）。
 *
 * 与早期版本的差别：早期用 `data/price.ts` 的 mock 数据 + 关键词匹配假装比价，
 * 本版接真实链路。因此**去掉了"近 6 个月价格走势"图**——后端没有历史价格数据源，
 * 画一条编造的曲线比不画更糟。
 *
 * 页面的核心设计取向：**诚实优先于好看**。
 * 这套比价基于公开搜索摘要而非电商官方价格 API，存在三类固有限制
 * （部分平台取不到价 / 价格可能属不同规格 / 有索引延迟），
 * 故结果区把"取不到的"和"被排除的"都明确列出来，而不是只展示几条漂亮的最低价。
 */
import { useCallback, useRef, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, LabelList, Tooltip, XAxis, YAxis } from 'recharts'
import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  ExternalLink,
  Image as ImageIcon,
  Info,
  Loader2,
  Package,
  Search,
  Sparkles,
  Trophy,
  Upload,
} from 'lucide-react'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Card from '../components/Card'
import PageHeader from '../components/PageHeader'
import Skeleton from '../components/Skeleton'
import { useToast } from '../components/Toast'
import { ApiError, comparePrice, type ComparePlatformResult, type CompareQuote, type PriceCompareResult } from '../lib/api'
import { ImagePrepareError, formatBytes, prepareImage, type PreparedImage } from '../lib/image'
import { LOW_COLOR, OTHER_COLOR, SUSPECT_COLOR, tooltipStyle } from '../lib/chartStyles'



/** 按可信度取徽章颜色与文案 */
function trustBadge(q: CompareQuote) {
  if (q.trust === 'extreme') return { color: 'danger' as const, text: '已排除' }
  if (q.trust === 'suspicious_high') return { color: 'warn' as const, text: '偏高，需核实' }
  if (q.trust === 'suspicious_low') return { color: 'warn' as const, text: '偏低，需核实' }
  return { color: 'ok' as const, text: '可信' }
}

function describeError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 413) return '图片过大，请压缩后重试'
    if (err.status === 400) return err.message
    if (err.status === 429) return '请求过于频繁，请稍后再试'
    if (err.status === 503) return '比价服务未就绪（后端未配置视觉模型或搜索密钥）'
    return `请求失败：${err.message}`
  }
  return '网络异常，请检查后端服务是否已启动'
}

/** 从各平台结果里挑出"该平台代表报价"：优先有价的、可信度最高的那条 */
function platformBestQuote(p: ComparePlatformResult): CompareQuote | null {
  const priced = p.quotes.filter((q) => q.has_price && q.price !== null)
  if (priced.length === 0) return p.quotes[0] ?? null
  // extreme 排到最后，normal 优先
  const rank = (q: CompareQuote) => (q.trust === 'normal' ? 0 : q.trust === 'extreme' ? 2 : 1)
  return [...priced].sort((a, b) => rank(a) - rank(b) || (a.price ?? 0) - (b.price ?? 0))[0]
}

export default function PriceCompare() {
  const toast = useToast()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [image, setImage] = useState<PreparedImage | null>(null)
  const [hint, setHint] = useState('')
  const [preparing, setPreparing] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PriceCompareResult | null>(null)
  const [error, setError] = useState('')

  const pickImage = useCallback(
    async (file: File | undefined) => {
      if (!file) return
      setPreparing(true)
      setError('')
      try {
        const prepared = await prepareImage(file)
        setImage(prepared)
        setResult(null)
      } catch (err) {
        toast.show(err instanceof ImagePrepareError ? err.message : '图片处理失败', 'error')
      } finally {
        setPreparing(false)
        if (fileInputRef.current) fileInputRef.current.value = ''
      }
    },
    [toast],
  )

  const run = useCallback(async () => {
    if (!image || loading) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await comparePrice(image.dataUrl, hint)
      setResult(res)
      if (!res.recommendation.winner) {
        toast.show('未能取到可比报价，请看下方提示', 'warning')
      } else {
        toast.show('比价完成', 'success')
      }
    } catch (err) {
      const msg = describeError(err)
      setError(msg)
      toast.show(msg, 'error')
    } finally {
      setLoading(false)
    }
  }, [image, hint, loading, toast])

  // ---- 结果派生数据 ----
  const pricedQuotes: CompareQuote[] = result
    ? result.platforms
        .map(platformBestQuote)
        .filter((q): q is CompareQuote => q !== null && q.has_price && q.price !== null)
    : []
  const winner = result?.recommendation.winner ?? null
  const comparable = pricedQuotes.filter((q) => q.trust !== 'extreme')
  const chartData = comparable
    .map((q) => ({ name: q.platform, price: q.price as number, trust: q.trust }))
    .sort((a, b) => a.price - b.price)
  const chartMax = chartData.length ? Math.max(...chartData.map((d) => d.price)) * 1.15 : 0

  return (
    <div>
      <PageHeader
        title="智能商品比价"
        description="上传商品图片 → 识别品牌型号 → 检索各平台公开报价 → 给出择优建议"
      />

      {/* ── 上传与检索 ─────────────────────────────────────────── */}
      <Card
        title="上传商品图片"
        subtitle="支持商品铭牌、包装、报价单截图；型号越清晰，比价越准"
        extra={
          image && (
            <Button
              variant="secondary"
              size="sm"
              icon={<Search className="h-3.5 w-3.5" />}
              loading={loading}
              disabled={preparing}
              onClick={() => void run()}
            >
              开始比价
            </Button>
          )
        }
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp,image/bmp"
          className="hidden"
          onChange={(e) => void pickImage(e.target.files?.[0])}
        />

        {!image ? (
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={preparing}
            className="flex w-full flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-ink-200 py-12 transition-colors hover:border-brand-300 hover:bg-brand-50/40 disabled:opacity-50"
          >
            {preparing ? (
              <Loader2 className="h-7 w-7 animate-spin text-brand-500" />
            ) : (
              <Upload className="h-7 w-7 text-ink-300" />
            )}
            <div className="text-[13px] font-medium text-ink-700">
              {preparing ? '正在处理图片…' : '点击选择商品图片'}
            </div>
            <div className="text-[11px] text-ink-400">
              自动压缩后上传，无需手动裁剪
            </div>
          </button>
        ) : (
          <div className="flex items-start gap-4">
            <img
              src={image.previewUrl}
              alt="商品图片"
              className="h-32 w-32 shrink-0 rounded-lg border border-slate-200 object-cover"
            />
            <div className="min-w-0 flex-1">
              <div className="truncate text-[13px] font-medium text-ink-900">{image.name}</div>
              <div className="mt-1 text-[11px] text-ink-400">
                {image.compressed
                  ? `已压缩 ${formatBytes(image.originalBytes)} → ${formatBytes(image.bytes)}`
                  : formatBytes(image.bytes)}
              </div>
              <div className="mt-3">
                <label className="text-[11px] text-ink-400">
                  补充说明（可选，帮助模型识别，如"这是包装盒侧面，型号是 A9F18220"）
                </label>
                <input
                  value={hint}
                  onChange={(e) => setHint(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && void run()}
                  className="mt-1 h-9 w-full rounded-lg border border-slate-200 px-3 text-[13px] text-ink-900 outline-none transition-colors focus:border-brand-500"
                  placeholder="可不填"
                />
              </div>
              <div className="mt-3 flex items-center gap-2">
                <Button
                  icon={<Search className="h-3.5 w-3.5" />}
                  loading={loading}
                  onClick={() => void run()}
                >
                  开始比价
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => {
                    setImage(null)
                    setResult(null)
                    setError('')
                  }}
                >
                  换一张
                </Button>
              </div>
            </div>
          </div>
        )}

        {loading && (
          <div className="mt-4 flex items-center gap-2 rounded-lg border border-brand-100 bg-brand-50 px-3 py-2 text-[12px] text-ink-600">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-brand-600" />
            正在识别商品并检索各平台报价，通常需要 5~20 秒…
          </div>
        )}

        {error && (
          <div className="mt-4 rounded-lg border-l-2 border-danger bg-red-50 px-4 py-3">
            <div className="flex items-center gap-2 text-[13px] font-semibold text-danger">
              <AlertTriangle className="h-4 w-4" />
              {error}
            </div>
          </div>
        )}
      </Card>

      {/* ── 加载骨架 ───────────────────────────────────────────── */}
      {loading && (
        <div className="mt-6 space-y-4">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-28 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      )}

      {!loading && result && (
        <>
          {/* 1. 识别出的商品 */}
          <Card
            className="mt-6"
            title="识别到的商品"
            subtitle={`检索关键词：${result.query}`}
            extra={
              <Badge color={result.product.confidence >= 0.8 ? 'ok' : 'warn'}>
                识别置信度 {(result.product.confidence * 100).toFixed(0)}%
              </Badge>
            }
          >
            <div className="flex items-center gap-5">
              <div className="flex h-20 w-20 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-brand-50 to-brand-100">
                <Package className="h-8 w-8 text-brand-500" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-base font-semibold text-ink-900">
                    {result.product.product_name || '（未识别出名称）'}
                  </span>
                  {result.product.brand && <Badge color="brand">品牌：{result.product.brand}</Badge>}
                  {result.product.category && <Badge color="neutral">{result.product.category}</Badge>}
                </div>
                <div className="mt-1.5 text-xs text-ink-500">
                  型号：<span className="font-mono text-ink-900">{result.product.model || '—'}</span>
                </div>
                <div className="mt-1 text-xs text-ink-500">规格：{result.product.spec || '—'}</div>
                {result.product.sku && (
                  <div className="mt-1 text-xs text-ink-500">
                    物料编码：<span className="font-mono">{result.product.sku}</span>
                  </div>
                )}
              </div>
              {winner && (
                <div className="shrink-0 text-right">
                  <div className="text-xs text-ink-400">最低报价</div>
                  <div className="mt-1 text-[28px] font-bold leading-8 text-ok tabular-nums">
                    ¥{winner.price?.toFixed(2)}
                  </div>
                  <div className="mt-0.5 text-[11px] text-ink-400">{winner.platform}</div>
                </div>
              )}
            </div>
          </Card>

          {/* 2. 推荐结论（含必读的风险提示） */}
          <div className="mt-4 rounded-xl border border-brand-100 bg-brand-50 p-5">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-brand-600" />
              <span className="text-[15px] font-semibold text-ink-900">比价建议</span>
              <Badge color="neutral">
                可比报价 {result.recommendation.priced_count} 条
              </Badge>
              {result.recommendation.excluded_count > 0 && (
                <Badge color="danger">
                  已排除 {result.recommendation.excluded_count} 条
                </Badge>
              )}
            </div>
            <p className="mt-2 text-[13px] leading-6 text-ink-700">
              {result.recommendation.summary || '未能在各平台取到可比报价。'}
            </p>
            {winner && (
              <a
                href={winner.url}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-3 inline-flex items-center gap-1 text-[12px] font-medium text-brand-700 underline"
              >
                <ExternalLink className="h-3 w-3" />
                打开 {winner.platform} 商品页核实价格
              </a>
            )}

            {result.recommendation.caveats.length > 0 && (
              <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3">
                <div className="flex items-center gap-1.5 text-[12px] font-semibold text-warn">
                  <Info className="h-3.5 w-3.5" />
                  使用前请知悉
                </div>
                <ul className="mt-1.5 space-y-1">
                  {result.recommendation.caveats.map((c, i) => (
                    <li key={i} className="flex gap-1.5 text-[11px] leading-5 text-ink-600">
                      <span className="text-warn">•</span>
                      <span>{c}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* 3. 平台报价卡 */}
          {comparable.length > 0 && (
            <>
              <h2 className="mb-3 mt-8 text-[15px] font-bold text-ink-900">
                平台报价（已排除非同型号）
              </h2>
              <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
                {[...comparable]
                  .sort((a, b) => (a.price ?? 0) - (b.price ?? 0))
                  .map((q) => {
                    const isWinner = winner && q.platform_key === winner.platform_key
                    const tb = trustBadge(q)
                    return (
                      <a
                        key={q.platform_key}
                        href={q.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={`relative block rounded-xl border p-4 transition-shadow hover:shadow-md ${
                          isWinner ? 'border-emerald-200 bg-emerald-50' : 'border-slate-200 bg-white'
                        }`}
                      >
                        {isWinner && (
                          <span className="absolute -right-2 -top-2 flex items-center gap-1 rounded-md bg-ok px-1.5 py-0.5 text-[10px] font-medium text-white">
                            <Trophy className="h-2.5 w-2.5" />
                            最优
                          </span>
                        )}
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-ink-500">{q.platform}</span>
                          <Badge color={q.kind === '工业品' ? 'brand' : 'neutral'}>{q.kind}</Badge>
                        </div>
                        <div
                          className={`mt-1.5 text-lg font-bold tabular-nums ${
                            isWinner ? 'text-ok' : 'text-ink-900'
                          }`}
                        >
                          ¥{q.price?.toFixed(2)}
                        </div>
                        <div className="mt-1.5">
                          <Badge color={tb.color}>{tb.text}</Badge>
                        </div>
                        <div className="mt-2 line-clamp-2 text-[11px] leading-4 text-ink-400" title={q.title}>
                          {q.title}
                        </div>
                        {q.note && (
                          <div className="mt-1.5 text-[10px] leading-4 text-warn">{q.note}</div>
                        )}
                      </a>
                    )
                  })}
              </div>
            </>
          )}

          {/* 4. 柱状图 */}
          {chartData.length > 1 && (
            <Card className="mt-4" title="平台价格对比" subtitle="绿色为最低报价">
              <BarChart width={900} height={260} data={chartData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                <XAxis
                  dataKey="name"
                  tickLine={false}
                  axisLine={{ stroke: '#E2E8F0' }}
                  tick={{ fontSize: 12, fill: '#64748B' }}
                />
                <YAxis
                  domain={[0, chartMax]}
                  tickFormatter={(v: number) => `¥${v}`}
                  tickLine={false}
                  axisLine={false}
                  tick={{ fontSize: 11, fill: '#94A3B8' }}
                  width={56}
                />
                <Tooltip
                  formatter={(value) => [`¥${Number(value).toFixed(2)}`, '报价']}
                  contentStyle={tooltipStyle}
                  cursor={{ fill: 'rgba(148, 163, 184, 0.08)' }}
                />
                <Bar dataKey="price" radius={[4, 4, 0, 0]} barSize={44} isAnimationActive={false}>
                  {chartData.map((d) => (
                    <Cell
                      key={d.name}
                      fill={
                        d.name === chartData[0].name
                          ? LOW_COLOR
                          : d.trust === 'normal'
                            ? OTHER_COLOR
                            : SUSPECT_COLOR
                      }
                    />
                  ))}
                  <LabelList
                    dataKey="price"
                    position="top"
                    formatter={(v: React.ReactNode) => `¥${Number(v).toFixed(2)}`}
                    style={{ fontSize: 11, fill: '#334155' }}
                  />
                </Bar>
              </BarChart>
              <div className="mt-2 flex flex-wrap items-center gap-4 text-[11px] text-ink-400">
                <span className="flex items-center gap-1">
                  <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: LOW_COLOR }} />
                  最低价
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: OTHER_COLOR }} />
                  正常报价
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: SUSPECT_COLOR }} />
                  需核实
                </span>
              </div>
            </Card>
          )}

          {/* 5. 各平台检索明细（诚实展示取不到与被排除的） */}
          <Card
            className="mt-4"
            title="各平台检索明细"
            subtitle="点开可看该平台取到的原始条目、被排除原因与商品页链接"
          >
            <div className="space-y-2">
              {result.platforms.map((p) => {
                const priced = p.quotes.filter((q) => q.has_price)
                const hasAny = priced.length > 0
                return (
                  <details key={p.platform_key} className="rounded-lg border border-slate-200">
                    <summary className="flex cursor-pointer flex-wrap items-center gap-2 px-4 py-2.5 text-[13px]">
                      {!p.ok ? (
                        <Ban className="h-3.5 w-3.5 text-danger" />
                      ) : hasAny ? (
                        <CheckCircle2 className="h-3.5 w-3.5 text-ok" />
                      ) : (
                        <AlertTriangle className="h-3.5 w-3.5 text-warn" />
                      )}
                      <span className="font-medium text-ink-900">{p.platform}</span>
                      <Badge color={p.kind === '工业品' ? 'brand' : 'neutral'}>{p.kind}</Badge>
                      <span className="text-[11px] text-ink-400">
                        同型号 {p.quotes.length} 条 · 含价 {priced.length} 条
                        {p.rejected_count > 0 && ` · 排除 ${p.rejected_count} 条`}
                        {!p.ok && ` · ${p.error}`}
                      </span>
                    </summary>
                    <div className="border-t border-slate-100 px-4 py-3">
                      {p.quotes.length > 0 ? (
                        <ul className="space-y-2">
                          {p.quotes.map((q, i) => (
                            <li key={i} className="flex items-start gap-2 text-[12px]">
                              <span className="w-20 shrink-0 tabular-nums text-ink-900">
                                {q.has_price ? `¥${q.price?.toFixed(2)}` : '无报价'}
                              </span>
                              <a
                                href={q.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="min-w-0 flex-1 truncate text-brand-700 underline"
                                title={q.title}
                              >
                                {q.title}
                              </a>
                              {q.trust !== 'normal' && <Badge color={trustBadge(q).color}>{trustBadge(q).text}</Badge>}
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <div className="text-[12px] text-ink-400">
                          {p.ok ? '该平台未搜到同型号商品' : p.error}
                        </div>
                      )}
                      {p.rejected.length > 0 && (
                        <div className="mt-3 border-t border-slate-100 pt-2">
                          <div className="flex items-center gap-1 text-[11px] font-medium text-ink-400">
                            <Ban className="h-3 w-3" />
                            已排除 {p.rejected_count} 条（型号或规格不符）
                          </div>
                          <ul className="mt-1 space-y-0.5">
                            {p.rejected.slice(0, 5).map((r, i) => (
                              <li key={i} className="flex gap-2 text-[11px] text-ink-400">
                                <span className="shrink-0 text-ink-300">[{r.reason}]</span>
                                <span className="min-w-0 truncate" title={r.title}>
                                  {r.title}
                                </span>
                              </li>
                            ))}
                            {p.rejected_count > 5 && (
                              <li className="text-[11px] text-ink-300">
                                另有 {p.rejected_count - 5} 条…
                              </li>
                            )}
                          </ul>
                        </div>
                      )}
                    </div>
                  </details>
                )
              })}
            </div>
          </Card>
        </>
      )}

      {/* ── 能力说明（未出结果时展示，让用户提前知道边界） ──────── */}
      {!result && !loading && !image && (
        <div className="mt-4 flex items-start gap-2 rounded-lg border border-slate-200 bg-white px-4 py-3 text-[11px] leading-5 text-ink-400">
          <ImageIcon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>
            本功能基于<b>公开搜索引擎的摘要</b>抓取各平台报价，不是电商官方价格接口。因此：
            京东、天猫等平台常因"需登录"取不到价；个别平台搜到的可能是不同规格商品，
            系统会按型号与规格过滤并标注可疑报价。最终价请以商品页为准。
          </span>
        </div>
      )}
    </div>
  )
}
