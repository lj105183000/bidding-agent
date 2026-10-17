import { describe, expect, it } from 'vitest'
import { isChunkLoadError } from './chunkError'

/**
 * 分块加载失败识别。
 *
 * ★ 为什么这些用例值得写：路由改成懒加载后，"发版 → 旧标签页点新页面"
 * 会走进这条分支。识别错的两种代价不对称：
 *   · 漏认（该认没认）→ 用户看到技术堆栈，以为系统坏了
 *   · 错认（不该认却认了）→ 用户被引导去"刷新"，而刷新并不能解决问题
 * 故这里把**各浏览器的真实文案**与**不该匹配的普通错误**都钉住。
 */
describe('isChunkLoadError', () => {
  it('识别 Chrome/Edge 的动态 import 失败文案', () => {
    expect(
      isChunkLoadError(
        'Failed to fetch dynamically imported module: http://127.0.0.1:5173/assets/Chat-a1b2c3.js',
      ),
    ).toBe(true)
  })

  it('识别 Firefox 文案', () => {
    expect(
      isChunkLoadError('error loading dynamically imported module: http://x/assets/Chat.js'),
    ).toBe(true)
  })

  it('识别 Safari 文案', () => {
    expect(isChunkLoadError('Importing a module script failed.')).toBe(true)
  })

  it('识别打包器/旧实现的 ChunkLoadError', () => {
    expect(isChunkLoadError('ChunkLoadError: Loading chunk 12 failed.')).toBe(true)
  })

  it('普通渲染错误不误判（否则会把用户引向无用的"刷新"）', () => {
    expect(isChunkLoadError('Cannot read properties of undefined (reading "sources")')).toBe(false)
    expect(isChunkLoadError('minified React error #130')).toBe(false)
    expect(isChunkLoadError('网络异常，请检查后端服务是否已启动')).toBe(false)
  })

  it('空值与非字符串安全返回 false', () => {
    expect(isChunkLoadError('')).toBe(false)
    expect(isChunkLoadError(undefined)).toBe(false)
    expect(isChunkLoadError(null)).toBe(false)
    expect(isChunkLoadError(404)).toBe(false)
    expect(isChunkLoadError(new Error('x'))).toBe(false)
  })

  it('大小写不敏感（不同浏览器大小写不一致）', () => {
    expect(isChunkLoadError('failed to fetch dynamically imported MODULE')).toBe(true)
    expect(isChunkLoadError('chunkloaderror')).toBe(true)
  })
})
