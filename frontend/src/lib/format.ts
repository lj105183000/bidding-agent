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
