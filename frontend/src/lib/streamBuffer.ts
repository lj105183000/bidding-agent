/**
 * 流式 token 缓冲：把同一帧内的多个 chunk 合并成一次提交。
 *
 * ★ 为什么要有这一层（实测，不是"优化习惯"）
 * ----------------------------------------
 * 后端一个 token 一个 SSE 帧，一次回答实测约 **240 帧**。原来每帧都
 * `setMessages` → 父组件重渲染 → 该条消息的 Markdown **整篇从头解析**。
 * 用 `react-dom/server` 真渲染实测（`logs/fe_markdown_bench.mjs`）：
 *
 *     每帧都解析全文      240 次渲染   760 ms
 *     合并到每 8 帧一次    30 次渲染    75 ms
 *     只在最后解析一次      1 次渲染     6 ms
 *
 * 合并的**上限**由屏幕刷新率决定（`requestAnimationFrame`）：一帧内无论来
 * 多少个 token 都只提交一次。人眼分辨不出"每帧更新"和"每 token 更新"，
 * 但 CPU 差好几倍。
 *
 * ★ 为什么做成纯逻辑（不直接在组件里写）
 * ---------------------------------
 * 这段逻辑有三个**容易写错**的地方，而它们都发生在"流结束的那一瞬间"，
 * 手点界面极难复现：
 *
 *   ① 最后一个 token 丢失 —— `done` 事件常与最后一个 token 同帧到达，
 *      若不先把缓冲提交就直接用 state 拼结果，末尾几个字会消失
 *   ② 中止后状态被覆盖 —— 用户点"停止"后要保留已生成内容并显示"已停止"，
 *      此时若缓冲照常提交并带上 `status:'streaming'`，提示会被覆盖掉
 *   ③ `reset` 之后迟到帧把气泡又填回来 —— 缓冲没跟着清空
 *
 * 做成**可注入调度器**的纯工厂后，三种情形都能用假时钟精确单测
 * （见 `streamBuffer.test.ts`），不需要浏览器。
 */

/** 一次提交的内容：正文增量 + 思考过程增量。 */
export interface FlushPayload {
  content: string
  thinking: string
}

export interface StreamBuffer {
  /** 追加正文 token（不立即提交，等下一帧）。 */
  pushToken(chunk: string): void
  /** 追加思考过程 token（与正文共用同一帧提交，避免渲染两次）。 */
  pushThinking(chunk: string): void
  /**
   * 立即提交缓冲中的内容。
   *
   * @returns 是否真的提交了（缓冲为空时返回 false，调用方可据此跳过 setState）
   */
  flush(): boolean
  /** 丢弃缓冲（用于 `reset`：后端要求清空已生成内容）。 */
  clear(): void
  /** 当前尚未提交的内容（只读，测试与调试用）。 */
  peek(): FlushPayload
  /** 取消已排期的提交（组件卸载时调用，避免对已卸载组件 setState）。 */
  cancel(): void
}

export interface StreamBufferOptions {
  /** 提交回调。**只有缓冲非空时**才会被调用。 */
  onFlush: (payload: FlushPayload) => void
  /** 调度器（默认 rAF）；测试注入假实现。 */
  schedule?: (cb: () => void) => number
  /** 取消调度（默认 cancelAnimationFrame）。 */
  cancelSchedule?: (handle: number) => void
}

export function createStreamBuffer(options: StreamBufferOptions): StreamBuffer {
  const schedule =
    options.schedule ??
    ((cb: () => void) => {
      // 无 rAF 的环境（Node / 老旧浏览器）退化为同步排队，行为仍正确
      if (typeof requestAnimationFrame === 'function') return requestAnimationFrame(cb)
      return setTimeout(cb, 0) as unknown as number
    })
  const cancelSchedule =
    options.cancelSchedule ??
    ((handle: number) => {
      if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(handle)
      else clearTimeout(handle as unknown as ReturnType<typeof setTimeout>)
    })

  let content = ''
  let thinking = ''
  let handle: number | null = null

  const arm = () => {
    if (handle !== null) return // 本帧已排期
    handle = schedule(() => {
      handle = null
      flush()
    })
  }

  const flush = (): boolean => {
    if (handle !== null) {
      cancelSchedule(handle)
      handle = null
    }
    if (!content && !thinking) return false // 空缓冲不触发 setState
    const payload = { content, thinking }
    content = ''
    thinking = ''
    options.onFlush(payload)
    return true
  }

  return {
    pushToken(chunk: string) {
      if (!chunk) return
      content += chunk
      arm()
    },
    pushThinking(chunk: string) {
      if (!chunk) return
      thinking += chunk
      arm()
    },
    flush,
    clear() {
      if (handle !== null) {
        cancelSchedule(handle)
        handle = null
      }
      content = ''
      thinking = ''
    },
    peek() {
      return { content, thinking }
    },
    cancel() {
      if (handle !== null) {
        cancelSchedule(handle)
        handle = null
      }
    },
  }
}
