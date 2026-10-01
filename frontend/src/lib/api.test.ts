/**
 * SSE 解析器最小单测（接口联调 spec §5：纯函数，不铺开测试框架）。
 * 覆盖：padding 注释/空行跳过、data: 前缀、[DONE] 尾帧、reset 事件、坏帧容错、中文保留。
 */
import { describe, it, expect } from 'vitest'
import { parseSse } from './api'

describe('parseSse', () => {
  it('跳过 2KB padding 注释行与空行', () => {
    const text = ': ' + ' '.repeat(2048) + '\n\ndata: {"type":"status","content":"x"}\n\n'
    const events = parseSse(text)
    expect(events).toHaveLength(1)
    expect(events[0].type).toBe('status')
  })

  it('只认 data: 前缀，忽略其他行', () => {
    const text = 'data: {"type":"token","content":"好"}\n\n随机行\n\nevent: foo\n\n'
    const events = parseSse(text)
    expect(events).toHaveLength(1)
    expect(events[0].type).toBe('token')
  })

  it('[DONE] 尾帧不产生事件', () => {
    const text = 'data: {"type":"done","elapsed_ms":5}\n\ndata: [DONE]\n\n'
    const events = parseSse(text)
    expect(events).toHaveLength(1)
    expect(events[0].type).toBe('done')
  })

  it('reset 事件正确解析', () => {
    const text = 'data: {"type":"reset"}\n\n'
    const events = parseSse(text)
    expect(events).toHaveLength(1)
    expect(events[0].type).toBe('reset')
  })

  it('坏帧 JSON 容错：跳过坏帧，不中断后续帧', () => {
    const text =
      'data: {"type":"token","content":"正常"}\n\n' +
      'data: 这不是JSON{{{bad\n\n' +
      'data: {"type":"token","content":"续"}\n\n'
    const events = parseSse(text)
    expect(events).toHaveLength(2)
    expect(events[0].content).toBe('正常')
    expect(events[1].content).toBe('续')
  })

  it('保留中文（ensure_ascii=False 口径）', () => {
    const events = parseSse('data: {"type":"token","content":"招投标采购Agent"}\n\n')
    expect(events[0].content).toBe('招投标采购Agent')
  })

  it('无 type 字段的帧被忽略', () => {
    const events = parseSse('data: {"content":"缺type"}\n\n')
    expect(events).toHaveLength(0)
  })

  it('done 帧保留 tool_name 单字符串与 phase_times', () => {
    const events = parseSse(
      'data: {"type":"done","sources":[],"tool_name":"search_knowledge_graph",' +
        '"phase_times":[["首轮分析",1200]]}\n\n',
    )
    expect(events[0].type).toBe('done')
    expect(events[0].tool_name).toBe('search_knowledge_graph')
    expect((events[0].phase_times as unknown[])[0]).toEqual(['首轮分析', 1200])
  })
})
