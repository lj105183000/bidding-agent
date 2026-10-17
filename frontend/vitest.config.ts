import { defineConfig } from 'vitest/config'

/**
 * 测试配置。
 *
 * ★ 关于 `include`：为什么把 `.tsx` 也收进来
 * ----------------------------------------
 * 原来只收 `*.test.ts`，于是"写了 `Foo.test.tsx` 却**静默不跑**"——
 * 用例文件存在、CI 全绿，实际一条没执行。加宽 glob 后这种文件至少会被
 * 收集到（若环境不支持会明确报错，而不是无声跳过）。
 *
 * ⚠️ DOM 测试需要额外依赖（**当前未安装**）
 * -------------------------------------
 * 环境是 `node`，没有 DOM。要测"渲染组件"（如消息气泡、追问卡片）需先装：
 *
 *     npm i -D jsdom @testing-library/react @testing-library/jest-dom
 *
 * 然后在需要 DOM 的文件顶部加一行 pragma（不必全局改环境，纯逻辑用例
 * 继续跑 node，更快）：
 *
 *     // @vitest-environment jsdom
 *
 * ★ 当前为什么不需要它（可测的部分已被抽成纯函数）
 * --------------------------------------------
 * 前端最容易错的两处逻辑已抽成不依赖 DOM 的纯模块并覆盖：
 *   · `lib/streamBuffer.ts`  流式 token 合并（末帧不丢 / 中止不覆盖状态 / reset 后不复活）
 *   · `lib/scroll.ts`        是否跟随滚动（用户上翻时不被拽回底部）
 * 另外 `lib/askStream.live.test.ts` 是**真后端**集成测试，默认跳过，
 * 需要时用 `LIVE_API=1 npm run test` 打开（验证真 SSE 分块下不丢字）。
 */
export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
