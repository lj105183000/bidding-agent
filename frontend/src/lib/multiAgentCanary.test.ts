import { describe, expect, it } from 'vitest'

import {
  DECISION_LABELS,
  MODE_LABELS,
  ROUTER_LABELS,
  formatMs,
  formatRate,
  hasMultiAgentData,
  rowsFromCounts,
} from './multiAgentCanary'

describe('rowsFromCounts', () => {
  it('maps known keys to labels and computes shares', () => {
    const rows = rowsFromCounts({ auto: 3, off: 1 }, MODE_LABELS)
    const auto = rows.find((row) => row.key === 'auto')
    expect(auto?.label).toBe('auto')
    expect(auto?.share).toBe(0.75)
  })

  it('keeps unknown keys readable', () => {
    const rows = rowsFromCounts({ future_mode: 2 }, MODE_LABELS)
    expect(rows[0].label).toBe('future_mode')
    expect(rows[0].count).toBe(2)
  })

  it('returns empty for missing counts', () => {
    expect(rowsFromCounts(undefined, MODE_LABELS)).toEqual([])
  })
})

describe('formatRate', () => {
  it('renders null as an em dash, not 0%', () => {
    expect(formatRate(null)).toBe('—')
    expect(formatRate(undefined)).toBe('—')
  })

  it('renders percentage with sample size', () => {
    expect(formatRate(0.5, 10)).toBe('50.0%（n=10）')
  })
})

describe('formatMs', () => {
  it('converts ms to seconds', () => {
    expect(formatMs(1500)).toBe('1.5s')
  })

  it('renders missing as an em dash', () => {
    expect(formatMs(null)).toBe('—')
  })
})

describe('hasMultiAgentData', () => {
  it('is false without a snapshot', () => {
    expect(hasMultiAgentData(undefined)).toBe(false)
  })

  it('is false when all counters are zero', () => {
    expect(hasMultiAgentData({ mode_counts: { auto: 0 } })).toBe(false)
  })

  it('is true when there is at least one sample', () => {
    expect(hasMultiAgentData({ mode_counts: { auto: 2 } })).toBe(true)
  })
})

describe('decision labels', () => {
  it('labels the auto-declined reasons in Chinese', () => {
    expect(DECISION_LABELS.declined_single_expert).toBe('单专家回退')
    expect(DECISION_LABELS.declined_short).toBe('短问题回退')
  })
})

describe('router labels', () => {
  it('maps router names to readable labels', () => {
    const rows = rowsFromCounts({ rules: 2, llm: 1 }, ROUTER_LABELS)
    expect(rows.find((row) => row.key === 'rules')?.label).toBe('规则')
    expect(rows.find((row) => row.key === 'llm')?.label).toBe('LLM')
  })
})
