import { Suspense, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { ChevronDown } from 'lucide-react'
import { NAV_ITEMS } from '../config/nav'
import { MODEL_OPTIONS, useModel } from '../contexts/model'
import BrandLogo from './BrandLogo'
import { MobileBottomNav, MobileNavDrawer } from './MobileNav'
import { PageLoading } from './RouteFallback'

export default function Layout() {
  const { model, setModel } = useModel()
  const location = useLocation()
  const [demoMode, setDemoMode] = useState(false)
  const [navOpen, setNavOpen] = useState(false)

  /** 当前页标题：给移动端顶栏用（窄屏没有侧栏，用户需要知道自己在哪一页）。 */
  const currentLabel =
    NAV_ITEMS.find((item) =>
      item.path === '/' ? location.pathname === '/' : location.pathname.startsWith(item.path),
    )?.label ?? ''

  return (
    /* ★ 根包裹用 `min-h-dvh` 而不是 `min-h-screen`
       ------------------------------------------
       `min-h-screen` = `100vh`。在**移动浏览器**上 `vh` 是**大视口**高度
       （地址栏收起时的值），比当前可见区域高出一截——实测在 714×450 的
       移动模拟下 `100vh` 解析成 **503px**，于是这个根 div 高 503px，
       **把文档撑到 503px**，整页又变成可滚动（正文内部滚动的意义就没了）。

       改成 `dvh`（dynamic viewport height）后根 div 与可见区一致，
       文档不再被撑高。同时保留 `min-h-screen` 作为不支持 `dvh` 的浏览器的回退
       （Tailwind 3.4+ 才有 `min-h-dvh`，旧版会静默不生成类名）。 */
    <div className="min-h-screen [@supports(height:100dvh)]:min-h-dvh">
      {/* 左侧固定侧边栏: <1024px **完全隐藏**（由底部导航 + 抽屉接管）
          
          ★ 为什么是 hidden 而不是收窄成 64px 图标栏
          ----------------------------------------
          原实现在窄屏把它压成 64px 纯图标栏。手机上这样有三处硬伤：
          13 个图标要竖着滚动、没有文字标签只能猜、且白占 17% 的横向空间。
          移动端的惯例是底部标签栏（拇指可达），故窄屏直接隐藏本组件。 */}
      <aside className="fixed inset-y-0 left-0 z-10 hidden w-[232px] flex-col border-r border-slate-200 bg-white lg:flex">
        {/* Logo 区 */}
        <div className="flex items-center gap-3 border-b border-slate-200 px-5 py-4">
          {/* 品牌 Logo。
              原来是一个渐变方块 + 通用的 Sparkles 图标 —— 那个图标在
              任何"AI 产品"里都长一样，无法传达"招投标采购"这个具体领域。
              换成自绘的「采购文件 + 法槌 + 智能星」见 public/logo.svg。
              Logo 自带蓝底圆角，故这里不再套渐变容器（否则双层底色）。 */}
          <BrandLogo className="h-8 w-8" />
          <div className="hidden min-w-0 lg:block">
            <div className="text-[15px] font-bold text-ink-900">招采智脑</div>
            <div className="truncate text-[10px] leading-tight text-ink-400">
              大模型驱动的数字智能招投标采购平台
            </div>
          </div>
        </div>

        {/* 导航菜单 */}
        <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4">
          {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              end={path === '/'}
              title={label}
              className={({ isActive }) =>
                `relative flex items-center justify-start gap-3 rounded-lg px-3 py-2.5 text-[13px] font-medium transition-colors ${
                  isActive
                    ? 'bg-brand-50 text-brand-700'
                    : 'text-ink-500 hover:bg-ink-50 hover:text-ink-700'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <span className="absolute -left-3 top-1/2 h-6 w-[3px] -translate-y-1/2 rounded-r bg-brand-600" />
                  )}
                  <Icon className="h-4 w-4 shrink-0" />
                  <span className="inline">{label}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* 演示模式开关: 切换全站等宽数字字体 */}
        <div className="border-t border-slate-200 px-4 py-3">
          <button
            onClick={() => setDemoMode((v) => !v)}
            className="flex w-full items-center justify-between"
          >
            <span className="text-[11px] text-ink-500">演示模式 · 等宽数字</span>
            <span
              className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${
                demoMode ? 'bg-ok' : 'bg-ink-200'
              }`}
            >
              <span
                className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${
                  demoMode ? 'left-[18px]' : 'left-0.5'
                }`}
              />
            </span>
          </button>
        </div>

        {/* 底部模型切换器（模拟） */}
        <div className="border-t border-slate-200 px-4 py-4">
          <div className="mb-2 text-[11px] text-ink-400">模型</div>
          <div className="flex rounded-lg bg-ink-50 p-1">
            {MODEL_OPTIONS.map((m) => (
              <button
                key={m}
                onClick={() => setModel(m)}
                className={`flex-1 rounded-md px-1 py-1.5 text-[11px] transition-all ${
                  model === m
                    ? 'bg-white font-medium text-brand-700 shadow-sm'
                    : 'text-ink-500 hover:text-ink-700'
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>
      </aside>

      {/* 移动端顶栏：显示当前页名 + 模型切换
          
          ★ 为什么移动端需要一个顶栏
          ------------------------
          窄屏隐藏侧栏后，"当前在哪一页"只剩下内容区自己说。
          数据密集页的首屏往往是表格/图表，标题可能在滚动区里，
          用户滑下去就失去了位置感。顶栏给一个常驻的页面标识。
          
          顶栏只放**当前页名**，不放返回按钮：底部导航是平级的，
          没有层级可返回，放个返回箭头反而让人以为有上级页面。 */}
      <header className="pt-safe sticky top-0 z-20 border-b border-slate-200 bg-white/95 backdrop-blur lg:hidden">
        <div className="flex items-center justify-between px-4 py-2.5">
          <div className="flex min-w-0 items-center gap-2">
            {/* 与侧栏同一个组件、同一份 SVG —— 两处品牌标识不会各自演化 */}
            <BrandLogo className="h-7 w-7" />
            <span className="truncate text-[14px] font-semibold text-ink-900">
              {currentLabel || '招采智脑'}
            </span>
          </div>

          {/* 模型切换收进一个紧凑下拉：
              移动端顶栏横向空间只够放一个控件，而模型切换是演示常用项 */}
          <label className="relative flex items-center">
            <span className="sr-only">选择模型</span>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value as (typeof MODEL_OPTIONS)[number])}
              className="touch-target appearance-none rounded-lg border border-slate-200 bg-white py-1.5 pl-2.5 pr-7 text-[12px] font-medium text-ink-700"
            >
              {MODEL_OPTIONS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-2 h-3.5 w-3.5 text-ink-400" />
          </label>
        </div>
      </header>

      {/* 移动端底部导航 + 抽屉 */}
      <MobileBottomNav onOpenMore={() => setNavOpen(true)} />
      <MobileNavDrawer open={navOpen} onClose={() => setNavOpen(false)} />

      {/* 右侧内容区: 页面切换 150ms 淡入, 演示模式下全站等宽数字

          ★ 桌面端（≥1024px）改为"整屏高的 flex 列 + 正文内部滚动"
          ------------------------------------------------------
          原来是 `min-h-screen` + 整页滚动：正文一长（或用户放大页面），
          下方内容就被推到视口外，**必须滚整页**才能看到——
          顶部区域（页头、推荐问题卡片）也跟着滚走，失去了位置感。

          现在改成与「智能问答」页同样的形态：
              main 固定为一屏高（flex 列，不自身滚动）
              └─ 上：页头等不滚动的部分（由页面自己决定）
              └─ 下：flex-1 + overflow-y-auto（正文滚动条）

          为什么用 flex 而不是 `h-[calc(100vh-140px)]`
          ----------------------------------------
          Chat 页那个 `calc(100vh-140px)` 是**估的常量**，实测不准：
              桌面：可用高度 = vh - 0(header) - 64(main padding)   → 应减 64，不是 140
              移动：可用高度 = vh - 53(header) - 54(bottomNav) - 80(padding)
          `calc` 还得区分断点各写一个魔数，改一处忘一处就错位。
          flex 让高度由布局自动分配 —— 加/减顶栏、改内边距都不需要重算。

          ★ `h-dvh` 而不是 `h-screen`
          -------------------------
          `h-screen` = `100vh`，在移动浏览器上是**最大**视口高度（地址栏收起时），
          比当前可见区域高出一截。实测在 714×450 的移动模拟下 `100vh` 解析成
          **503px** —— 于是 main 高 503px，**把文档撑到 503px**，整页又变成可滚动，
          正文内部滚动的意义就没了（第一次改完就是这个结果，由探针量出来）。

          全部改用 `dvh`（dynamic viewport height）后，main 与可见区一致。
          保留 `h-screen` 作为不支持 `dvh` 的浏览器的回退
          （Tailwind 3.4+ 才有 `h-dvh`，旧版会静默不生成类名）。
          同一 layer 内按类名排序，`h-dvh` 在后故覆盖 `h-screen`。

          ★ 为什么不再限制在 lg（但要给所有页面一层滚动容器）
          --------------------------------------------------
          第一版只在 `lg`（≥1024px）启用，结果**用户放大页面时正好失效**：
          浏览器缩放到 200% 会让 CSS 像素视口宽度降到 720px，跨过断点退回整页滚动，
          而"放大后被挤出屏幕"恰恰是本次要修的问题（实测确认）。

          但全断点启用后跑全站回归发现**移动端 5 个页面 main 溢出**：
          `dvh` 在移动端解析为**大视口**高度（实测 897px），比可见区高，
          而 main 被固定成这个高度后，内容比它更长就溢出。

          真正的问题是：**固定 main 高度后，"页面内容比视口长"这件事必须有
          一层滚动容器来承接**，否则内容被裁掉。而原来只有首页与智能问答页
          各自处理了滚动，其他 11 个页面靠的是整页滚动——约束一加就没了兜底。

          故这一版给**所有页面**统一提供一层 `flex-1 overflow-y-auto` 的滚动容器：
              内容比视口短  → 不出现滚动条（常态，视觉无变化）
              内容比视口长  → 自动出内部滚动条，内容不会被裁

          `min-h-0` 必须加在链路每一层：flex 子项默认 `min-height:auto`，
          会被内容撑高，滚动条就不出现了（这是 flex 布局最常踩的坑）。

          ★ 与智能问答页的配合
          -----------------
          该页自己也是 flex 列且内部有滚动区，会自然填满这层容器；
          它原来的 `lg:h-[calc(100vh-140px)]`（估的常量）已改为 `lg:h-full`
          ——父级高度现在由布局正确分配，不需要再猜。

          ★ 收尾：这道滚动容器同时解决了"main 溢出"与"内容被裁"两个问题，
          比给 11 个页面各加一次 `overflow-y-auto` 更不易漏。 */}
      <main className="flex h-screen flex-col px-4 py-4 pb-nav-safe [@supports(height:100dvh)]:h-dvh sm:px-6 sm:py-6 lg:ml-[232px] lg:p-8 lg:pb-8">
        <div
          className={`flex min-h-0 flex-1 flex-col ${
            demoMode ? 'demo-tabular' : ''
          }`}
        >
          {/* key 放在这一层：路由变化时重放淡入动画，同时保持外层不重建
              （外层重建会让内部滚动位置与 flex 布局抖动）。
              滚动容器与 route key 放在同一层，切换路由时滚动位置自动归零。

              ★ Suspense 也放在这一层（页面已改为路由级懒加载）
              ----------------------------------------------
              占位块必须落在**滚动容器内部**：否则它会撑在 main 外面，
              与"内容区滚动"的布局约束打架。而挂在 `<Outlet/>` 外面
              还有一个好处 —— 等待页面 chunk 时**侧栏、移动端顶栏、
              底部导航都保持挂载**，用户可以直接切去别的页面，
              而不是面对一个整屏 loading（实测 `Chat` 页 chunk ≈200 kB，
              首次进入确实有可感知的等待）。 */}
          <div
            key={location.pathname}
            className="page-fade flex min-h-0 flex-1 flex-col overflow-y-auto"
          >
            <Suspense fallback={<PageLoading />}>
              <Outlet />
            </Suspense>
          </div>
        </div>
      </main>
    </div>
  )
}
