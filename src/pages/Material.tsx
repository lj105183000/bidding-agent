import { Fragment, useState } from 'react'
import {
  ArrowRight,
  ChevronRight,
  Layers,
  Package,
  ScanSearch,
  Target,
  TrendingDown,
  UserCheck,
} from 'lucide-react'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Card from '../components/Card'
import EmptyState from '../components/EmptyState'
import PageHeader from '../components/PageHeader'
import Skeleton from '../components/Skeleton'
import { useToast } from '../components/Toast'
import { MATERIAL_GROUPS, type MaterialGroup } from '../data/material'

const DEFAULT_TEXT = '波斯 加长球头内六角扳手 BS423181 1/16"-3/8" 1套'

/** 描述关键词 → 物料组映射 */
const MATCH_RULES: [string, string[]][] = [
  ['mat-001', ['内六角', 'BS423181', '波斯', '扳手']],
  ['mat-002', ['断路器', 'iC65N', '施耐德', '微断']],
  ['mat-003', ['口罩', '9502', '3M', 'KN95']],
]

function matchMaterial(text: string): MaterialGroup | null {
  const hit = MATCH_RULES.find(([, kws]) => kws.some((k) => text.includes(k)))
  return hit ? MATERIAL_GROUPS.find((g) => g.id === hit[0])! : null
}

function scoreColor(s: number): 'ok' | 'brand' | 'warn' {
  if (s >= 98) return 'ok'
  if (s >= 95) return 'brand'
  return 'warn'
}

function sourceColor(s: string): 'brand' | 'ok' | 'warn' {
  if (s === 'ERP') return 'brand'
  if (s === 'MES') return 'ok'
  return 'warn'
}

const GOVERNANCE = [
  { label: '物料记录去重率', value: '38.6%', icon: Layers, cls: 'bg-brand-50 text-brand-600' },
  { label: '编码准确率', value: '96.2%', icon: Target, cls: 'bg-emerald-50 text-ok' },
  { label: '采购成本下降', value: '7.4%', icon: TrendingDown, cls: 'bg-amber-50 text-warn' },
  { label: '人工审核工作量下降', value: '62%', icon: UserCheck, cls: 'bg-red-50 text-danger' },
]

export default function Material() {
  const toast = useToast()
  const [text, setText] = useState(DEFAULT_TEXT)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<MaterialGroup | null>(null)

  const recognize = () => {
    const t = text.trim()
    if (!t) {
      toast.show('请输入物料描述', 'warning')
      return
    }
    setLoading(true)
    setResult(null)
    setTimeout(() => {
      const hit = matchMaterial(t)
      setResult(hit)
      setLoading(false)
      if (!hit) toast.show('未识别到匹配的标准物料，请补充型号或品牌信息', 'warning')
    }, 800)
  }

  return (
    <div>
      <PageHeader title="物料智能识别" description="一物多码识别与物料标准化" />

      {/* 第一部分: 单条识别 */}
      <div className="flex items-start gap-4">
        <Card className="flex-1" title="单条物料识别" subtitle="输入物料描述，AI 识别标准编码">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="h-40 w-full resize-none rounded-lg border border-slate-200 p-3 text-[13px] leading-6 text-ink-900 outline-none transition-colors focus:border-brand-500"
            placeholder="输入物料描述，如：波斯 加长球头内六角扳手…"
          />
          <div className="mt-4">
            <Button
              onClick={recognize}
              loading={loading}
              icon={<ScanSearch className="h-3.5 w-3.5" />}
            >
              智能识别
            </Button>
          </div>
        </Card>

        <Card className="flex-1" title="识别结果" subtitle="标准编码 · 名称 · 分类 · 规格 · 品牌">
          {loading && (
            <div className="space-y-3">
              <Skeleton className="h-5 w-2/3" />
              <Skeleton className="h-5 w-1/3" />
              <Skeleton className="h-6 w-full" />
              <Skeleton className="h-5 w-1/2" />
              <Skeleton className="h-5 w-2/5" />
            </div>
          )}

          {!loading && !result && (
            <EmptyState
              icon={<Package className="h-6 w-6" />}
              title="暂无识别结果"
              description='输入物料描述后点击"智能识别"，AI 将匹配标准物料编码'
            />
          )}

          {!loading && result && (
            <div className="space-y-4">
              <div>
                <div className="text-[11px] text-ink-400">标准编码</div>
                <div className="mt-1 font-mono text-[15px] font-semibold text-brand-700">
                  {result.stdCode}
                </div>
              </div>

              <div>
                <div className="text-[11px] text-ink-400">标准名称</div>
                <div className="mt-1 text-[15px] font-semibold text-ink-900">{result.name}</div>
              </div>

              <div>
                <div className="text-[11px] text-ink-400">分类路径</div>
                <div className="mt-1 flex flex-wrap items-center gap-1">
                  {result.categoryPath.split(' > ').map((seg, i) => (
                    <Fragment key={`${seg}-${i}`}>
                      {i > 0 && <ChevronRight className="h-3 w-3 text-ink-300" />}
                      <Badge color={i === result.categoryPath.split(' > ').length - 1 ? 'brand' : 'neutral'}>
                        {seg}
                      </Badge>
                    </Fragment>
                  ))}
                </div>
              </div>

              <div>
                <div className="text-[11px] text-ink-400">规格参数</div>
                <div className="mt-1 text-[13px] text-ink-700">{result.spec}</div>
              </div>

              <div>
                <div className="text-[11px] text-ink-400">品牌</div>
                <div className="mt-1">
                  <Badge color="neutral">{result.brand}</Badge>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between text-[11px] text-ink-400">
                  <span>匹配度</span>
                  <span className="tabular-nums text-ink-700">{result.matchScore}%</span>
                </div>
                <div className="mt-1.5 h-1.5 rounded-full bg-ink-100">
                  <div
                    className={`h-full rounded-full ${
                      result.matchScore >= 98 ? 'bg-ok' : result.matchScore >= 95 ? 'bg-brand-500' : 'bg-warn'
                    }`}
                    style={{ width: `${result.matchScore}%` }}
                  />
                </div>
              </div>
            </div>
          )}
        </Card>
      </div>

      {/* 第二部分: 一物多码归并 */}
      <Card
        className="mt-6"
        title="一物多码归并"
        subtitle="AI 将来自 ERP / MES / SRM 的异构描述归并为统一标准编码"
      >
        <div className="space-y-4">
          {MATERIAL_GROUPS.map((g) => (
            <div key={g.id} className="rounded-xl border border-slate-200 p-5">
              {/* 顶部: 编码 + 名称 + 匹配度 */}
              <div className="flex items-center gap-3">
                <span className="rounded-md bg-ink-900 px-2 py-1 font-mono text-xs text-white">
                  {g.stdCode}
                </span>
                <span className="text-[15px] font-semibold text-ink-900">{g.name}</span>
                <Badge color={scoreColor(g.matchScore)}>匹配度 {g.matchScore}%</Badge>
              </div>

              {/* 中间: 左原始描述 → 右归并结果 */}
              <div className="mt-4 flex items-center gap-4">
                <div className="min-w-0 flex-1 space-y-2">
                  {g.rawDescriptions.map((d) => (
                    <div key={d.source} className="flex items-center gap-2">
                      <Badge color={sourceColor(d.source)}>{d.source}</Badge>
                      <span className="truncate text-xs text-ink-700" title={d.text}>
                        {d.text}
                      </span>
                    </div>
                  ))}
                </div>
                <ArrowRight className="h-5 w-5 shrink-0 text-brand-600" />
                <div className="min-w-0 flex-1 rounded-lg border border-brand-100 bg-brand-50 p-3">
                  <div className="text-xs text-ink-500">归并结果</div>
                  <div className="mt-1 text-[13px] font-semibold text-ink-900">{g.name}</div>
                  <div className="mt-0.5 font-mono text-xs text-brand-700">{g.stdCode}</div>
                  <div className="mt-1 text-[11px] leading-4 text-ink-500">{g.categoryPath}</div>
                  <div className="mt-1 text-[11px] text-ink-500">
                    规格：{g.spec} ｜ 品牌：{g.brand}
                  </div>
                </div>
              </div>

              {/* 底部: 归并统计 */}
              <div className="mt-3 text-xs text-ink-400">
                {g.rawDescriptions.length} 条记录 → 1 个标准编码
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* 第三部分: 治理成效 */}
      <h2 className="mb-4 mt-8 text-[15px] font-bold text-ink-900">治理成效</h2>
      <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
        {GOVERNANCE.map((k) => (
          <Card key={k.label}>
            <div className="flex items-center gap-3">
              <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${k.cls}`}>
                <k.icon className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <div className="text-[22px] font-bold leading-7 text-ink-900 tabular-nums">
                  {k.value}
                </div>
                <div className="mt-0.5 text-xs text-ink-400">{k.label}</div>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}
