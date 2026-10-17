import { describe, expect, it, vi } from 'vitest'
import { createStreamBuffer, type FlushPayload } from './streamBuffer'

/**
 * ★ 这些用例覆盖的是"流结束那一瞬间"的三个易错点。
 *
 * 它们在手工点界面时**极难复现**（要么丢字、要么提示被覆盖、要么清空后复活），
 * 但都是真实发生过的失败形态，故用假调度器把时间线固定下来测。
 */
function makeBuffer() {
  const flushes: FlushPayload[] = []
  // 假调度器：把回调存起来，由用例显式"推进一帧"
  const scheduled = new Map<number, () => void>()
  let nextId = 1
  const buffer = createStreamBuffer({
    onFlush: (p) => flushes.push(p),
    schedule: (cb) => {
      const id = nextId++
      scheduled.set(id, cb)
      return id
    },
    cancelSchedule: (id) => {
      scheduled.delete(id)
    },
  })
  const tick = () => {
    const pending = [...scheduled.entries()]
    scheduled.clear()
    pending.forEach(([, cb]) => cb())
  }
  return { buffer, flushes, tick, scheduled }
}

describe('createStreamBuffer', () => {
  it('同一帧内的多个 token 只提交一次（合并成一次渲染）', () => {
    const { buffer, flushes, tick } = makeBuffer()
    buffer.pushToken('依据')
    buffer.pushToken('《招标')
    buffer.pushToken('投标法》')
    expect(flushes).toHaveLength(0) // 未到帧边界不提交
    tick()
    expect(flushes).toEqual([{ content: '依据《招标投标法》', thinking: '' }])
  })

  it('跨帧的 token 分别提交，且顺序不乱', () => {
    const { buffer, flushes, tick } = makeBuffer()
    buffer.pushToken('A')
    tick()
    buffer.pushToken('B')
    tick()
    expect(flushes.map((f) => f.content)).toEqual(['A', 'B'])
  })

  it('正文与思考过程共用一个提交（不额外触发一次渲染）', () => {
    const { buffer, flushes, tick } = makeBuffer()
    buffer.pushThinking('先查法规') // 思考在前
    buffer.pushToken('答案是 2%') // 同一帧内来正文
    tick()
    expect(flushes).toEqual([{ content: '答案是 2%', thinking: '先查法规' }])
  })

  it('★ 末帧不丢：flush() 立即提交尚未到帧边界的 token', () => {
    const { buffer, flushes } = makeBuffer()
    buffer.pushToken('最后')
    buffer.pushToken('几个字')
    // 模拟 done 事件与末帧 token 同帧到达
    const did = buffer.flush()
    expect(did).toBe(true)
    expect(flushes).toEqual([{ content: '最后几个字', thinking: '' }])
  })

  it('★ 空缓冲 flush 不触发提交（不产生多余的 setState）', () => {
    const { buffer, flushes } = makeBuffer()
    expect(buffer.flush()).toBe(false)
    expect(flushes).toHaveLength(0)
  })

  it('★ flush 会撤销已排期的提交，避免重复提交', () => {
    const { buffer, flushes, tick, scheduled } = makeBuffer()
    buffer.pushToken('内容')
    expect(scheduled.size).toBe(1)
    buffer.flush()
    expect(scheduled.size).toBe(0) // 排期已被取消
    tick() // 再推进一帧也不该产生新的提交
    expect(flushes).toHaveLength(1)
  })

  it('★ clear() 丢弃缓冲：reset 之后迟到的排期提交不会把气泡填回来', () => {
    const { buffer, flushes, tick, scheduled } = makeBuffer()
    buffer.pushToken('要被丢弃的内容')
    buffer.clear()
    expect(buffer.peek()).toEqual({ content: '', thinking: '' })
    expect(scheduled.size).toBe(0)
    tick()
    expect(flushes).toHaveLength(0) // 不能"复活"
  })

  it('clear() 之后新 token 仍能正常提交', () => {
    const { buffer, flushes, tick } = makeBuffer()
    buffer.pushToken('旧')
    buffer.clear()
    buffer.pushToken('新')
    tick()
    expect(flushes).toEqual([{ content: '新', thinking: '' }])
  })

  it('cancel() 阻止后续提交（组件卸载/流结束后的排期不再打到状态上）', () => {
    const { buffer, flushes, tick } = makeBuffer()
    buffer.pushToken('迟到')
    buffer.cancel()
    tick()
    expect(flushes).toHaveLength(0)
    // 但 cancel 不丢内容：显式 flush 仍能拿到
    expect(buffer.peek().content).toBe('迟到')
  })

  it('空字符串 token 不触发排期（SSE 心跳帧常见）', () => {
    const { buffer, flushes, tick, scheduled } = makeBuffer()
    buffer.pushToken('')
    buffer.pushThinking('')
    expect(scheduled.size).toBe(0)
    tick()
    expect(flushes).toHaveLength(0)
  })

  it('peek 不改动缓冲（供 keepStatus 路径读取后再 clear）', () => {
    const { buffer, flushes } = makeBuffer()
    buffer.pushToken('保留内容')
    expect(buffer.peek().content).toBe('保留内容')
    expect(buffer.peek().content).toBe('保留内容')
    expect(flushes).toHaveLength(0)
  })

  it('没有 rAF 的环境也能工作（默认调度器退化路径）', () => {
    // 该用例只验"不抛异常且最终会提交"，不依赖真实计时
    const onFlush = vi.fn()
    const buffer = createStreamBuffer({
      onFlush,
      schedule: (cb) => {
        cb()
        return 1
      },
      cancelSchedule: () => {},
    })
    buffer.pushToken('x')
    expect(onFlush).toHaveBeenCalledWith({ content: 'x', thinking: '' })
  })
})
