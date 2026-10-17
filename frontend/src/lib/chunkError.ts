/**
 * 识别"懒加载分块取不到"这一类错误。
 *
 * ★ 为什么值得单独识别（页面改懒加载之后新增的失败形态）
 * -------------------------------------------------
 * 用户开着旧标签页，期间前端发版 → html 里记的旧 chunk 名（`Chat-a1b2c3.js`）
 * 在服务器上已被新构建取代 → 点进该页面时动态 import 失败。
 *
 * 这不是用户操作错误、也不是程序 bug：**刷新一下就好**。但如果按普通
 * "页面加载出错"提示（还带一段技术堆栈），用户会以为系统坏了。
 *
 * ★ 为什么用正则匹配消息而不是看错误类型
 * ----------------------------------
 * 同一个语义在不同浏览器里抛出的类型与文案都不同：
 *     Chrome/Edge   TypeError: Failed to fetch dynamically imported module: http://…
 *     Firefox       TypeError: error loading dynamically imported module: http://…
 *     Safari        Importing a module script failed.
 *     打包器/旧实现  ChunkLoadError
 * 没有统一的 `error.code` 可用，故按文案识别 —— 这是社区通行做法，
 * 也意味着**匹配不上时只是回退到通用提示**，不会误伤（宁可少认，不可错认）。
 */
const CHUNK_ERROR_PATTERNS = [
  /Failed to fetch dynamically imported module/i,
  /error loading dynamically imported module/i,
  /Importing a module script failed/i,
  /ChunkLoadError/i,
]

export function isChunkLoadError(message: unknown): boolean {
  if (typeof message !== 'string' || !message) return false
  return CHUNK_ERROR_PATTERNS.some((re) => re.test(message))
}
