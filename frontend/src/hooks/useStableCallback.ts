import { useCallback, useEffect, useRef } from 'react'

/**
 * 返回一个**引用恒定**、但每次调用都走**最新实现**的回调。
 *
 * ★ 为什么需要它（实测：一次流式回答里历史消息被重渲染 ~200 次）
 * ------------------------------------------------------------
 * 聊天页每收到一个 token 就 `setMessages`，于是所有依赖 `messages` 的
 * `useCallback` 都会换新引用。把这种回调当 props 传给子组件时，
 * **即使子组件套了 `React.memo` 也照样全部重渲染** —— memo 是浅比较，
 * 有一个 prop 变了就失效。实测（`logs/fe_markdown_bench.mjs`）：
 * 每次重渲染都要把整条消息的 Markdown **从头解析一遍**，
 * 240 个 token 累计 760 ms CPU，且这是"每条消息各来一遍"。
 *
 * 两种天真的修法都有问题：
 *   · 直接给 `memo` 写自定义比较器忽略函数 props → 子组件会**闭包过期**
 *     （`submitClarification` 里 `messages.findIndex(...)` 读到旧快照，
 *      追问卡就会找不到对应的问题）
 *   · 把回调的依赖清空 → 同样是闭包过期，只是更隐蔽
 *
 * 故这里用"ref 保存最新实现 + 恒定外壳"的写法：**引用永远不变**（memo 生效），
 * 调用时读 `ref.current`（永远是最新闭包，**不会过期**）。
 *
 * ⚠️ 赋值放在 `useEffect` 里而不是渲染期间：并发渲染下渲染期写 ref
 * 可能写入一个"被丢弃的那次渲染"的闭包。放在 effect 里保证只提交已生效的实现。
 */
export default function useStableCallback<A extends unknown[], R>(
  fn: (...args: A) => R,
): (...args: A) => R {
  const ref = useRef(fn)
  useEffect(() => {
    ref.current = fn
  })
  return useCallback((...args: A) => ref.current(...args), [])
}
