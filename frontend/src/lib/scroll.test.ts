import { describe, expect, it } from 'vitest'
import { FOLLOW_THRESHOLD_PX, isNearBottom } from './scroll'

/**
 * 滚动跟随判定。
 *
 * ★ 修掉的行为：原来流式时无条件 `scrollTop = scrollHeight`，
 * 用户往上翻看上文会被每个 token 拽回底部。故这里把"是否跟随"抽成纯函数，
 * 用边界值把语义固定下来。
 */
describe('isNearBottom', () => {
  it('正好在底部 → 跟随', () => {
    expect(isNearBottom({ scrollHeight: 1000, scrollTop: 800, clientHeight: 200 })).toBe(true)
  })

  it('距离底部小于阈值 → 仍跟随（滚轮/触控板常差几个像素）', () => {
    const gap = FOLLOW_THRESHOLD_PX - 1
    expect(
      isNearBottom({ scrollHeight: 1000, scrollTop: 800 - gap, clientHeight: 200 }),
    ).toBe(true)
  })

  it('距离底部超过阈值 → 不跟随（用户正在回看上文）', () => {
    const gap = FOLLOW_THRESHOLD_PX + 1
    expect(
      isNearBottom({ scrollHeight: 1000, scrollTop: 800 - gap, clientHeight: 200 }),
    ).toBe(false)
  })

  it('内容还没撑满容器（无滚动条）→ 视为在底部', () => {
    expect(isNearBottom({ scrollHeight: 120, scrollTop: 0, clientHeight: 200 })).toBe(true)
  })

  it('内容不足一屏但被轻微滚动 → 仍在底部', () => {
    expect(isNearBottom({ scrollHeight: 200, scrollTop: 0, clientHeight: 200 })).toBe(true)
  })

  it('浮点滚动位置不会误判（浏览器 scrollTop 可以是小数）', () => {
    expect(isNearBottom({ scrollHeight: 1234.5, scrollTop: 1034.5, clientHeight: 200 })).toBe(true)
  })

  it('阈值可覆盖（便于以后按屏幕尺寸调整）', () => {
    const metrics = { scrollHeight: 1000, scrollTop: 700, clientHeight: 200 } // 距底 100
    expect(isNearBottom(metrics)).toBe(false) // 默认 80
    expect(isNearBottom(metrics, 120)).toBe(true) // 放宽到 120
  })
})
