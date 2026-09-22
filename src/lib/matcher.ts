import { QA_DATA, type QAItem } from '../data/qa'

export interface QAMatch {
  item: QAItem
  score: number
}

const STOP_WORDS = new Set(['什么', '怎么', '如何', '哪些', '请问', '是否', '可以', '的话', '以及'])

/** 按标点切分后取 2~4 字滑动窗口作为关键词, 过滤停用词 */
function tokenize(input: string): string[] {
  const tokens = input
    .split(/[\s,，。？?、；;：:（）()《》“”"']+/)
    .map((t) => t.trim())
    .filter((t) => t.length >= 2)
  const grams = new Set<string>()
  for (const t of tokens) {
    if (STOP_WORDS.has(t)) continue
    const max = Math.min(4, t.length)
    for (let len = 2; len <= max; len++) {
      for (let i = 0; i + len <= t.length; i++) {
        grams.add(t.slice(i, i + len))
      }
    }
  }
  return [...grams]
}

/**
 * 简单中文关键词匹配:
 * - 关键词命中 question 得 2 分
 * - 关键词命中 tags 得 2 分 + 标签额外 +1 分
 * 按分数降序返回, 过滤 0 分
 */
export function matchQA(question: string): QAMatch[] {
  const input = question.trim()
  if (!input) return []
  const grams = tokenize(input)
  const results: QAMatch[] = []
  for (const item of QA_DATA) {
    let score = 0
    for (const g of grams) {
      if (item.question.includes(g)) score += 2
      if (item.tags.some((t) => t === g || t.includes(g) || g.includes(t))) score += 3
    }
    if (score > 0) results.push({ item, score })
  }
  return results.sort((a, b) => b.score - a.score)
}
