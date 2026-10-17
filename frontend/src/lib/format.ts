/** 金额格式化: 千分位 + ¥ 符号, 保留 2 位小数 */
export function formatMoney(n: number): string {
  return `¥${n.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

/** 中文金额: >=1亿 用亿元, >=1万 用万元, 否则用元 */
export function formatMoneyCN(n: number): string {
  if (n >= 1e8) return `${(n / 1e8).toFixed(2)} 亿元`
  if (n >= 1e4) return `${(n / 1e4).toFixed(2)} 万元`
  return `${n.toFixed(2)} 元`
}

/** 日期串归一化: '2023-10-13 18:17:16' / ISO 字符串 → '2023-10-13' */
export function shortDate(s: string): string {
  const m = /^(\d{4})[-/](\d{1,2})[-/](\d{1,2})/.exec(s)
  if (m) return `${m[1]}-${m[2].padStart(2, '0')}-${m[3].padStart(2, '0')}`
  const d = new Date(s)
  if (Number.isNaN(d.getTime())) return s
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

/**
 * 毫秒 → 可读秒数。
 *
 * ★ 为什么 <1000ms 时给 2 位小数
 * -------------------------
 * `0.4s` 与 `0.04s` 差一个数量级，但 `0.4s`/`0.0s`（1 位）看不出区别。
 * ≥1s 时 1 位小数够用（`12.3s`），再多是噪声。
 *
 * ★ 为什么收编到 `lib/format.ts`
 * ------------------------
 * 原先它只定义在 `Chat.tsx` 里（未导出）。RAG 耗时面板也需要同样的格式，
 * 若各自实现一遍，同一次耗时在两个面板里会显示成"1.2s"与"1.20s" ——
 * 而用户会以为它们不一致。与 `chartStyles.ts` 同一考虑。
 */
export function fmtSec(ms?: number | null): string {
  if (ms === undefined || ms === null || Number.isNaN(ms)) return '—'
  const s = ms / 1000
  return s < 1 ? `${s.toFixed(2)}s` : `${s.toFixed(1)}s`
}
