import { useEffect, useState } from 'react'
import { CheckCircle2, Copy, Cpu, Download, FileSearch, Loader2 } from 'lucide-react'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Card from '../components/Card'
import PageHeader from '../components/PageHeader'
import { useToast } from '../components/Toast'
import { useModel } from '../contexts/model'
import { EXTRACT_SAMPLES, type ExpectedFields } from '../data/extractSamples'
import { formatMoney, shortDate } from '../lib/format'

type ExtractedResult = ExpectedFields
type Status = 'idle' | 'extracting' | 'done'

interface FieldDef {
  key: keyof ExtractedResult
  label: string
}

/** 17 个字段, 顺序与验收标准一致 */
const FIELDS: FieldDef[] = [
  { key: 'title', label: '标题' },
  { key: 'category', label: '类别' },
  { key: 'source', label: '来源' },
  { key: 'publishTime', label: '发布时间' },
  { key: 'province', label: '省份' },
  { key: 'city', label: '市区' },
  { key: 'county', label: '县城' },
  { key: 'projectNo', label: '项目编号' },
  { key: 'projectName', label: '项目名称' },
  { key: 'purchaser', label: '采购人' },
  { key: 'agency', label: '代理机构' },
  { key: 'budget', label: '预算' },
  { key: 'address', label: '项目地址' },
  { key: 'period', label: '周期' },
  { key: 'winner', label: '中标人' },
  { key: 'winAmount', label: '中标金额' },
  { key: 'winTime', label: '中标时间' },
]

const TAB_NAMES = ['示例一', '示例二', '示例三']

function categoryColor(c: string): 'brand' | 'ok' | 'warn' | 'neutral' {
  if (c === '货物') return 'brand'
  if (c === '服务') return 'ok'
  if (c === '工程') return 'warn'
  return 'neutral'
}

function sourceColor(s: string): 'brand' | 'ok' | 'warn' | 'neutral' {
  if (s === '政府采购') return 'brand'
  if (s === '企业采购') return 'ok'
  if (s === '招标采购') return 'warn'
  return 'neutral'
}

function FieldValue({ field, value }: { field: FieldDef; value: string | number }) {
  if (value === '') return <span className="text-ink-300">—</span>
  if (field.key === 'category') return <Badge color={categoryColor(String(value))}>{value}</Badge>
  if (field.key === 'source') return <Badge color={sourceColor(String(value))}>{value}</Badge>
  if (field.key === 'winAmount')
    return <span className="font-medium tabular-nums">{formatMoney(Number(value))}</span>
  if (field.key === 'publishTime' || field.key === 'winTime') return shortDate(String(value))
  return <span className="break-words">{String(value)}</span>
}

export default function Extract() {
  const toast = useToast()
  const { model } = useModel()
  const [active, setActive] = useState(0)
  const [input, setInput] = useState(EXTRACT_SAMPLES[0].rawText)
  const [status, setStatus] = useState<Status>('idle')
  const [revealed, setRevealed] = useState(0)
  const [result, setResult] = useState<ExtractedResult | null>(null)

  // 提取时每 90ms 填充一个字段
  useEffect(() => {
    if (status !== 'extracting') return
    if (revealed >= FIELDS.length) {
      setStatus('done')
      return
    }
    const t = setTimeout(() => setRevealed((r) => r + 1), 90)
    return () => clearTimeout(t)
  }, [status, revealed])

  const selectSample = (i: number) => {
    setActive(i)
    setInput(EXTRACT_SAMPLES[i].rawText)
    setStatus('idle')
    setRevealed(0)
    setResult(null)
  }

  const startExtract = () => {
    if (!input.trim()) {
      toast.show('请先输入或选择公告文本', 'warning')
      return
    }
    setStatus('extracting')
    setRevealed(0)
    setResult(EXTRACT_SAMPLES[active].expected)
  }

  const copyJson = () => {
    if (!result) return
    navigator.clipboard
      .writeText(JSON.stringify(result, null, 2))
      .then(() => toast.show('JSON 已复制到剪贴板', 'success'))
      .catch(() => toast.show('复制失败', 'error'))
  }

  const downloadCsv = () => {
    if (!result) return
    const rows = [['字段', '值'], ...FIELDS.map((f) => [f.label, String(result[f.key])])]
    const csv =
      '﻿' +
      rows.map((r) => r.map((c) => `"${c.replace(/"/g, '""')}"`).join(',')).join('\r\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `标讯提取结果_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(url)
    toast.show('CSV 已下载', 'success')
  }

  const statusBar = (
    <div className="flex items-center gap-2">
      {status === 'idle' && (
        <>
          <span className="h-2 w-2 rounded-full bg-ink-300" />
          <span className="text-[13px] text-ink-500">等待提取</span>
        </>
      )}
      {status === 'extracting' && (
        <>
          <Loader2 className="h-3.5 w-3.5 animate-spin text-brand-600" />
          <span className="text-[13px] text-brand-700">
            正在解析，逐字段提取中…（{revealed}/{FIELDS.length}）
          </span>
        </>
      )}
      {status === 'done' && (
        <>
          <CheckCircle2 className="h-3.5 w-3.5 text-ok" />
          <span className="text-[13px] text-ok">提取完成 · 共 {FIELDS.length} 个字段</span>
        </>
      )}
    </div>
  )

  return (
    <div>
      <PageHeader title="标讯智能提取" description="从招标公告中提取结构化标讯信息（17 字段）" />

      <div className="flex items-start gap-4">
        {/* 左栏: 输入 */}
        <Card className="flex-1" title="公告原文" subtitle="选择示例或粘贴公告文本">
          <div className="mb-3 flex gap-2">
            {EXTRACT_SAMPLES.map((s, i) => (
              <button
                key={s.id}
                onClick={() => selectSample(i)}
                title={s.name}
                className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                  active === i
                    ? 'border-brand-200 bg-brand-50 text-brand-700'
                    : 'border-slate-200 text-ink-500 hover:bg-ink-50 hover:text-ink-700'
                }`}
              >
                {TAB_NAMES[i]}
              </button>
            ))}
          </div>

          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="h-[520px] w-full resize-none rounded-lg border border-slate-200 p-3 text-[13px] leading-6 text-ink-900 outline-none transition-colors focus:border-brand-500"
            placeholder="在此粘贴招标公告原文…"
          />

          <div className="mt-4 flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-xs text-ink-400">
              <Cpu className="h-3.5 w-3.5" />
              当前模型：{model}
            </div>
            <Button
              onClick={startExtract}
              loading={status === 'extracting'}
              icon={<FileSearch className="h-3.5 w-3.5" />}
            >
              开始智能提取
            </Button>
          </div>
        </Card>

        {/* 右栏: 输出 */}
        <Card className="flex-1" title="结构化提取结果" subtitle="17 个标准字段" extra={statusBar}>
          <div className="grid grid-cols-2 gap-3">
            {FIELDS.map((f, i) => {
              const filled = i < revealed
              const isCurrent = status === 'extracting' && i === revealed
              return (
                <div
                  key={f.key}
                  className={`rounded-lg border px-3 py-2.5 transition-colors ${
                    isCurrent
                      ? 'animate-pulse border-brand-200 bg-brand-50'
                      : filled
                        ? 'border-slate-200 bg-white'
                        : 'border-slate-100 bg-ink-50'
                  }`}
                >
                  <div className="text-[11px] text-ink-400">{f.label}</div>
                  <div className="mt-1 text-[13px] leading-5 text-ink-900">
                    {filled || isCurrent ? (
                      <FieldValue field={f} value={result?.[f.key] ?? ''} />
                    ) : (
                      <span className="text-ink-200">· · ·</span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>

          <div className="mt-4 flex justify-end gap-2">
            <Button
              variant="secondary"
              size="sm"
              icon={<Copy className="h-3.5 w-3.5" />}
              disabled={status !== 'done'}
              onClick={copyJson}
            >
              复制 JSON
            </Button>
            <Button
              variant="secondary"
              size="sm"
              icon={<Download className="h-3.5 w-3.5" />}
              disabled={status !== 'done'}
              onClick={downloadCsv}
            >
              下载 CSV
            </Button>
            <Button variant="ghost" size="sm" disabled={status === 'extracting'} onClick={startExtract}>
              重新提取
            </Button>
          </div>
        </Card>
      </div>
    </div>
  )
}
