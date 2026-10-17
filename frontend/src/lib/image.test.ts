/**
 * 图片预处理工具的单测（只测纯函数与可推导的分支）。
 *
 * 为什么不断言 prepareImage 的压缩结果：它依赖 DOM 的 Image / canvas / FileReader，
 * vitest 的 node 环境没有这些对象（项目也未引入 jsdom），强行 mock 一整套 DOM
 * 只会测出 mock 的行为，而不是真实压缩结果。压缩效果由浏览器端人工验证覆盖。
 * 这里守住的是**体积换算**——它决定了"会不会撞上后端 4MB 上限"的判断。
 */
import { describe, it, expect } from 'vitest'
import { formatBytes } from './image'

describe('formatBytes', () => {
  it('小于 1KB 用 B', () => {
    expect(formatBytes(0)).toBe('0 B')
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(1023)).toBe('1023 B')
  })

  it('小于 1MB 用 KB（取整）', () => {
    expect(formatBytes(1024)).toBe('1 KB')
    expect(formatBytes(2048)).toBe('2 KB')
    expect(formatBytes(300 * 1024)).toBe('300 KB')
  })

  it('1MB 及以上用 MB（一位小数）', () => {
    expect(formatBytes(1024 * 1024)).toBe('1.0 MB')
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB')
    // 3.7MB 这类常见手机照片体积
    expect(formatBytes(Math.round(3.7 * 1024 * 1024))).toBe('3.7 MB')
  })

  it('与后端上限可比：4,000,000 base64 字符 ≈ 3MB 二进制', () => {
    // base64 膨胀约 4/3，故 4e6 字符 ≈ 3e6 字节 ≈ 2.9MB
    const approxBytes = Math.floor((4_000_000 * 3) / 4)
    expect(formatBytes(approxBytes)).toBe('2.9 MB')
  })
})
