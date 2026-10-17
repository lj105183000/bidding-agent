import { useEffect, useState } from 'react'

/**
 * 订阅一个媒体查询，返回它当前是否匹配。
 *
 * ★ 为什么需要 JS 而不是纯 CSS
 * -------------------------
 * 纯 CSS 断点只能控制**样式**。而移动端与桌面端的差异不止样式：
 *
 *     底部导航 vs 侧边栏      —— 是两套不同的 DOM 结构，不是同一元素的样式切换
 *     抽屉的开合状态           —— 需要 state
 *     切换路由后自动关抽屉     —— 需要副作用
 *     力导向图的触控手势       —— 需要知道当前是不是触屏（决定是否启用拖拽/缩放）
 *
 * 这些都必须知道"当前是不是移动端"。用 `matchMedia` 而不是监听 `resize`：
 * 前者只在**跨越断点**时触发，后者每次拖动窗口都会触发（手机旋转、
 * 地址栏收放都会触发 resize，用它会导致大量无意义重渲染）。
 *
 * 断点与 Tailwind 的 `lg` 保持一致（1024px）——若两者不一致，
 * 会出现"CSS 认为是桌面布局、JS 认为是移动端"的错位，非常难排查。
 */
export const MOBILE_BREAKPOINT = 1024

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => {
    // SSR / 测试环境没有 matchMedia：返回 false（按桌面处理，最保守）
    if (typeof window === 'undefined' || !window.matchMedia) return false
    return window.matchMedia(query).matches
  })

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return
    const mql = window.matchMedia(query)
    const onChange = (e: MediaQueryListEvent) => setMatches(e.matches)
    // 先同步一次：首次渲染与 effect 之间用户可能已经改过窗口大小
    setMatches(mql.matches)
    // 现代浏览器用 addEventListener；旧 Safari 只有 addListener
    if (mql.addEventListener) {
      mql.addEventListener('change', onChange)
      return () => mql.removeEventListener('change', onChange)
    }
    mql.addListener(onChange)
    return () => mql.removeListener(onChange)
  }, [query])

  return matches
}

/** 是否移动端布局（宽度 < 1024px，与 Tailwind 的 lg 断点一致）。 */
export function useIsMobile(): boolean {
  return useMediaQuery(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`)
}

/**
 * 是否触屏设备。
 *
 * ★ 与"是否窄屏"是**两个维度**，不能混用
 * ------------------------------------
 *   窄屏 + 触屏   手机
 *   宽屏 + 触屏   平板、触屏笔记本
 *   窄屏 + 鼠标   桌面上把窗口拖窄
 *
 * 力导向图该不该启用触控手势、hover 提示要不要禁用，
 * 取决于是不是**触屏**，而不是窗口窄不窄。
 * 用一个 `isMobile` 同时管两件事，会在触屏笔记本上得到错误行为。
 */
export function useIsTouch(): boolean {
  return useMediaQuery('(pointer: coarse)')
}
