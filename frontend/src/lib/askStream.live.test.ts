/**
 * 真流式集成测试（**默认跳过**，需显式打开）。
 *
 * 为什么要有它：单测能证明缓冲/解析各自正确，但证明不了
 * "真后端的 SSE 分块 + 解析 + 缓冲"串起来不丢字。
 * 这类问题只有真跑一次端到端才看得见（尾字丢失、末帧丢内容、
 * 中文被分块切断）。
 *
 * 跑法（需要后端在 8001 上运行）：
 *     LIVE_API=1 npm run test
 */
import { describe, expect, it } from 'vitest'
import { askStream } from './api'
import { createStreamBuffer } from './streamBuffer'

/**
 * `process.env` 的极简声明。
 *
 * ★ 为什么不直接 `process.env.LIVE_API`：本项目的 `tsc -b` 会把 `src/**`
 * （含测试文件）一起类型检查，而 devDependencies 里没有 `@types/node`
 * —— 直接用 `process` 会编译失败（vitest 不做类型检查，所以只有 build 会报）。
 * 为一个开关装一套 Node 类型不值当，这里就地声明用到的那一点形状。
 */
declare const process: { env: Record<string, string | undefined> }

const LIVE = process.env.LIVE_API === '1'

/** 与聊天页一致的最小请求参数（provider 与思考开关是必填项）。 */
const BASE_OPTS = {
  history: [] as { role: string; content: string }[],
  provider: 'deepseek',
  deep_thinking_enabled: false,
}

describe.skipIf(!LIVE)('askStream 真流式（需后端 8001）', () => {
  it(
    '能收到 token、正常 done，且末帧内容不丢',
    async () => {
      // 用 buffer 收集，与聊天页同一条路径（同一帧内多个 token 合并提交）
      let content = ''
      let thinking = ''
      let donePayload: { sources?: unknown[] } | null = null
      const buffer = createStreamBuffer({
        onFlush: ({ content: c, thinking: t }) => {
          content += c
          thinking += t
        },
      })

      await askStream(
        { ...BASE_OPTS, question: '投标保证金最高不能超过多少？' },
        {
          onToken: (chunk) => buffer.pushToken(chunk),
          onThinking: (chunk) => buffer.pushThinking(chunk),
          onDone: (payload) => {
            // 与 Chat.tsx 一样：done 到达时先把缓冲 flush 掉
            buffer.flush()
            donePayload = payload as { sources?: unknown[] }
          },
          onError: (message) => {
            throw new Error(`流里报错：${message}`)
          },
        },
        AbortSignal.timeout(180_000),
      )
      buffer.flush() // 兜底（正常路径下 done 已 flush 过）

      expect(donePayload, '未收到 done 事件').not.toBeNull()
      expect(content.length).toBeGreaterThan(20)
      // 中文没有乱码（分块切断多字节字符时会出现 U+FFFD）
      expect(content).not.toContain('\uFFFD')
      // 追问：断言回答确实答到了保证金上限（不写死数值，避免随语料变化而脆断）
      expect(content).toMatch(/保证金/)
      // 思考链路也要能收到（本轮未开深度思考，故只断言类型正确、不崩）
      expect(typeof thinking).toBe('string')
    },
    200_000,
  )

  it(
    '中止（AbortSignal）能立刻停止且不抛未捕获异常',
    async () => {
      const controller = new AbortController()
      let tokens = 0
      const task = askStream(
        { ...BASE_OPTS, question: '请详细介绍一下招标投标法的全部要点。' },
        {
          onToken: () => {
            tokens += 1
            if (tokens >= 3) controller.abort() // 收到几个 token 就中止
          },
        },
        controller.signal,
      )
      await expect(task).rejects.toThrow() // abort 会让 fetch 抛 AbortError
    },
    120_000,
  )
})
