import { describe, expect, it } from 'vitest'

import {
  buildCompareGroups,
  changeLabel,
  changeTone,
  filterChangedRows,
  formatDelta,
  pickDefaultPair,
  statusChangeLabel,
  storageLabel,
} from './evalCompare'
import type { EvalCompareRow, EvalRunSummary } from './evalApi'

function run(runId: string): EvalRunSummary {
  return {
    run_id: runId,
    created_at: null,
    version: 'v1',
    schema_version: 1,
    source: 'cli',
    job_id: null,
    suites: ['agent'],
    suites_source: 'observed',
    verdict: 'pass',
    case_counts: { agent: 5 },
    suite_verdicts: { agent: 'pass' },
    metric_count: 1,
  }
}

function row(overrides: Partial<EvalCompareRow> = {}): EvalCompareRow {
  return {
    metric: 'agent.tool_accuracy',
    label: '工具选择准确率',
    suite: 'agent',
    unit: 'ratio',
    direction: 'higher_is_better',
    is_ratio: true,
    base: { value: 0.9, status: 'pass', threshold: 0.9 },
    head: { value: 0.85, status: 'fail', threshold: 0.9 },
    delta: -0.05,
    pct_delta: -0.0556,
    status_changed: true,
    status_change: 'pass->fail',
    change: 'regressed',
    track_regression: true,
    note: '',
    ...overrides,
  }
}

describe('pickDefaultPair', () => {
  it('少于两次运行无法对比', () => {
    expect(pickDefaultPair([])).toBeNull()
    expect(pickDefaultPair([run('r1')])).toBeNull()
  })

  it('默认 head=最新、base=上一次（后端已按时间倒序）', () => {
    expect(pickDefaultPair([run('r2'), run('r1')])).toEqual({
      base: 'r1',
      head: 'r2',
    })
  })
})

describe('formatDelta', () => {
  it('比例指标用百分点，不制造相对百分比歧义', () => {
    expect(formatDelta(-0.05, true)).toBe('-5.0 pp')
    expect(formatDelta(0.0123, true)).toBe('+1.2 pp')
    expect(formatDelta(0, true)).toBe('0.0 pp')
  })

  it('非比例指标用原单位带符号', () => {
    expect(formatDelta(1200, false)).toBe('+1,200')
    expect(formatDelta(-0.5, false)).toBe('-0.5')
    expect(formatDelta(0, false)).toBe('0')
  })

  it('null/NaN 显示破折号而不是 0', () => {
    expect(formatDelta(null, true)).toBe('—')
    expect(formatDelta(Number.NaN, false)).toBe('—')
  })
})

describe('状态与判定文案', () => {
  it('状态变化给出中文前后态', () => {
    expect(statusChangeLabel(row())).toBe('达标 → 未达标')
    expect(statusChangeLabel(row({ status_changed: false }))).toBe('')
  })

  it('change 文案与色调', () => {
    expect(changeLabel('improved')).toBe('进步')
    expect(changeLabel('regressed')).toBe('退化')
    expect(changeLabel('observed')).toBe('仅观测')
    expect(changeTone('improved')).toBe('good')
    expect(changeTone('regressed')).toBe('bad')
    expect(changeTone('unchanged')).toBe('neutral')
  })
})

describe('分组与过滤', () => {
  it('按 suite 分组并保留顺序', () => {
    const rows = [
      row({ metric: 'agent.a', suite: 'agent' }),
      row({ metric: 'rag.b', suite: 'rag' }),
      row({ metric: 'agent.c', suite: 'agent' }),
    ]
    const groups = buildCompareGroups(rows)
    expect(groups.map(([suite]) => suite)).toEqual(['agent', 'rag'])
    expect(groups[0][1].map((item) => item.metric)).toEqual(['agent.a', 'agent.c'])
  })

  it('只看有变化：状态变化或数值非零变化', () => {
    const rows = [
      row({ metric: 'a' }),
      row({ metric: 'b', status_changed: false, delta: 0 }),
      row({ metric: 'c', status_changed: false, delta: 0.001 }),
    ]
    expect(filterChangedRows(rows).map((item) => item.metric)).toEqual(['a', 'c'])
  })
})

describe('storageLabel', () => {
  it('如实标注数据库与回退来源', () => {
    expect(
      storageLabel({
        backend: 'db',
        degraded: false,
        reason: '',
        db_available: true,
        file_available: true,
      }),
    ).toBe('数据库')
    expect(
      storageLabel({
        backend: 'trend_file',
        degraded: true,
        reason: 'db_empty',
        db_available: true,
        file_available: true,
      }),
    ).toContain('待回填')
    expect(
      storageLabel({
        backend: 'none',
        degraded: true,
        reason: 'db_unavailable',
        db_available: false,
        file_available: false,
      }),
    ).toBe('无数据')
  })
})
