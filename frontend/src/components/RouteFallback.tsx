/**
 * 路由兜底页 + 全局错误边界。
 *
 * 为什么要加这个
 * ------------
 * React Router 在**没有匹配路由**或**页面渲染抛错**时，会显示它自带的开发者
 * 错误页（"Unexpected Application Error! 404 Not Found / 💿 Hey developer 👋"）。
 * 那个页面：
 *   - 面向开发者，用户看不懂
 *   - **不告诉你哪个路径错了**，也不列出可用路径，只能靠猜
 *   - 在界面上直接暴露技术细节
 *
 * 故这里提供两件事：
 *   1. `NotFound`：未匹配路径时显示**实际访问的路径** + 全部可用页面的跳转链接
 *   2. `RouteError`：页面渲染抛错时给出可读提示与重试入口，而不是堆栈
 *
 * 实际访问的路径是关键信息——用户常是点了一个旧书签、或手输路径打错一个字，
 * 看到"404"却不知道错在哪，只能反复试。
 */
import { Link, isRouteErrorResponse, useLocation, useRouteError } from 'react-router-dom'
import { AlertTriangle, Compass, Home, RefreshCw } from 'lucide-react'
import Button from './Button'
import Card from './Card'
import Skeleton from './Skeleton'
import { NAV_ITEMS } from '../config/nav'
import { isChunkLoadError } from '../lib/chunkError'

/**
 * 页面懒加载时的占位（路由级 `Suspense` 的 fallback）。
 *
 * ★ 为什么需要它（而不是用空 div 或转圈）
 * ------------------------------------
 * 路由被拆包后，首次进入某个页面要等它的 chunk 下载完（实测最大的
 * `Chat` 页 chunk ≈200 kB）。这段时间若界面是**空白**，用户会以为点错了、
 * 或者页面卡住了；若用转圈，则会在内容出现时发生"跳一下"的布局位移。
 *
 * 故这里画一个**与真实页面结构相近的骨架**（页头 + 卡片），
 * 让占位块的位置与最终内容大致重合，切换时的位移最小。
 * `aria-busy` 让读屏软件知道这里还在加载（而不是读到一个空页面）。
 *
 * ⚠️ 刻意不做成"整屏 loading"：`Suspense` 挂在 `Layout` 的 `<Outlet/>` 外层，
 * 侧栏 / 移动端顶栏**保持可点**，用户在等待时可以直接切去别的页面。
 */
export function PageLoading() {
  return (
    <div className="flex flex-col gap-4 p-1" aria-busy="true" aria-live="polite">
      <span className="sr-only">页面加载中</span>
      <Skeleton className="h-7 w-48" />
      <Skeleton className="h-4 w-72" />
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} className="h-24 w-full" />
        ))}
      </div>
      <Skeleton className="h-40 w-full" />
    </div>
  )
}

/** 未匹配任何路由 */
export function NotFound() {
  const location = useLocation()

  return (
    <div className="flex min-h-[60vh] items-center justify-center p-6">
      <Card className="w-full max-w-2xl">
        <div className="flex items-start gap-3">
          <Compass className="mt-0.5 h-6 w-6 shrink-0 text-warn" />
          <div className="min-w-0 flex-1">
            <h1 className="text-[17px] font-bold text-ink-900">找不到这个页面</h1>
            <p className="mt-1.5 text-[13px] leading-6 text-ink-600">
              你访问的路径是：
              <code className="mx-1 rounded bg-ink-50 px-1.5 py-0.5 font-mono text-[12px] text-danger">
                {location.pathname}
              </code>
              ——它不在本系统的页面列表里。
            </p>
            <p className="mt-1 text-[12px] leading-5 text-ink-400">
              常见原因：旧书签、手输路径打错、或链接来自更早的版本。
            </p>

            <div className="mt-4">
              <div className="mb-2 text-[12px] font-medium text-ink-600">可用的页面：</div>
              <div className="flex flex-wrap gap-2">
                {NAV_ITEMS.map((item) => {
                  const Icon = item.icon
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-2.5 py-1.5 text-[12px] text-ink-700 transition-colors hover:border-brand-300 hover:bg-brand-50/50 hover:text-brand-700"
                    >
                      <Icon className="h-3.5 w-3.5" />
                      {item.label}
                      <code className="text-[10px] text-ink-400">{item.path}</code>
                    </Link>
                  )
                })}
              </div>
            </div>

            <div className="mt-5 flex items-center gap-2">
              <Link to="/">
                <Button icon={<Home className="h-3.5 w-3.5" />}>回到工作台</Button>
              </Link>
              <Button
                variant="secondary"
                onClick={() => window.location.reload()}
                icon={<RefreshCw className="h-3.5 w-3.5" />}
              >
                刷新页面
              </Button>
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}

/** 页面渲染抛错时的错误边界（替换 React Router 自带的开发者错误页） */
export function RouteError() {
  const error = useRouteError()
  const location = useLocation()

  // 区分"路由 404"与"页面抛异常"——两者的处理方式不同
  const is404 = isRouteErrorResponse(error) && error.status === 404
  const status = isRouteErrorResponse(error) ? error.status : null
  const detail = isRouteErrorResponse(error)
    ? error.statusText || ''
    : error instanceof Error
      ? error.message
      : String(error ?? '')

  /* ★ 分块加载失败：页面已改为路由级懒加载，这是它**特有的**一种失败
     --------------------------------------------------------------
     场景：用户开着旧标签页，期间前端**发了新版本**。旧页面的 html 里记的是
     旧的文件名（`Chat-a1b2c3.js`），而服务器上只剩新构建的 `Chat-d4e5f6.js`
     —— 点进"智能问答"时那个旧 chunk 已经不存在，动态 import 失败。

     这不是用户操作错误、也不是程序 bug：**刷新一下就好了**。
     而默认的"页面加载出错 + 一堆错误堆栈"会让人以为系统坏了，
     于是反复点、或者来报障（发版后这类问题会集中出现）。

     故识别出这种错误后，单独给一句人话 + 一个"刷新页面"按钮，
     技术细节仍保留在下面（便于排查，但不再是主角）。
     识别规则在 `lib/chunkError.ts`（纯函数，带各浏览器真实文案的单测）。

     ⚠️ 刻意**不**自动刷新：自动刷新会丢掉用户正在输入的内容
     （聊天页输入框、抽取页已选的文件），代价大于收益。 */
  const chunkStale = !is404 && isChunkLoadError(detail)

  return (
    <div className="flex min-h-[60vh] items-center justify-center p-6">
      <Card className="w-full max-w-2xl">
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-6 w-6 shrink-0 text-danger" />
          <div className="min-w-0 flex-1">
            <h1 className="text-[17px] font-bold text-ink-900">
              {is404 ? '找不到这个页面' : chunkStale ? '页面版本已更新' : '页面加载出错'}
            </h1>
            <p className="mt-1.5 text-[13px] leading-6 text-ink-600">
              {is404
                ? '该路径不在本系统的页面列表里。'
                : chunkStale
                  ? '这个页面的资源在你打开本页之后更新过，旧资源已不存在。刷新一下即可，其余功能不受影响。'
                  : '这个页面在渲染时抛出了异常，其余功能不受影响。'}
            </p>

            {chunkStale && (
              <div className="mt-3">
                <Button
                  icon={<RefreshCw className="h-3.5 w-3.5" />}
                  onClick={() => window.location.reload()}
                >
                  刷新页面
                </Button>
              </div>
            )}

            <div className="mt-2 space-y-1 rounded-lg bg-ink-50 px-3 py-2 text-[11px] leading-5">
              <div>
                <span className="text-ink-400">访问路径：</span>
                <code className="font-mono text-ink-700">{location.pathname}</code>
              </div>
              {status !== null && (
                <div>
                  <span className="text-ink-400">状态码：</span>
                  <code className="font-mono text-ink-700">{status}</code>
                </div>
              )}
              {detail && (
                <div className="break-words">
                  <span className="text-ink-400">错误信息：</span>
                  <span className="text-ink-700">{detail}</span>
                </div>
              )}
            </div>

            <div className="mt-4">
              <div className="mb-2 text-[12px] font-medium text-ink-600">
                {is404 ? '可用的页面：' : '试试这些入口：'}
              </div>
              <div className="flex flex-wrap gap-2">
                {NAV_ITEMS.map((item) => {
                  const Icon = item.icon
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-2.5 py-1.5 text-[12px] text-ink-700 transition-colors hover:border-brand-300 hover:bg-brand-50/50 hover:text-brand-700"
                    >
                      <Icon className="h-3.5 w-3.5" />
                      {item.label}
                    </Link>
                  )
                })}
              </div>
            </div>

            <div className="mt-5 flex items-center gap-2">
              <Link to="/">
                <Button icon={<Home className="h-3.5 w-3.5" />}>回到工作台</Button>
              </Link>
              <Button
                variant="secondary"
                onClick={() => window.location.reload()}
                icon={<RefreshCw className="h-3.5 w-3.5" />}
              >
                刷新页面
              </Button>
            </div>

            {!is404 && (
              <p className="mt-3 text-[11px] leading-5 text-ink-400">
                若反复出错，请把上面的「访问路径」与「错误信息」提供给开发者；
                后端服务与数据不受影响。
              </p>
            )}
          </div>
        </div>
      </Card>
    </div>
  )
}
