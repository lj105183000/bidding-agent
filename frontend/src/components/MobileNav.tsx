import { useEffect } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { Menu, X } from 'lucide-react'
import { NAV_ITEMS } from '../config/nav'

/**
 * 移动端导航：底部标签栏 + 全屏抽屉。
 *
 * ★ 为什么不用"把侧边栏压成 64px 图标栏"
 * ------------------------------------
 * 原实现在 <1024px 时把侧栏收成 64px 宽的**纯图标栏**。桌面端窗口拖窄时
 * 这样处理没问题，但在手机上它有三个硬伤：
 *
 *   1. **13 个图标竖排在 64px 宽里**，每屏只能看到 8~9 个，必须滚动
 *   2. **没有文字标签**，只靠图标猜功能（"变更追踪"与"招标文件解析"
 *      的图标在 16px 下几乎无法区分），违反移动端可识别性原则
 *   3. 占掉屏幕**左侧 17% 的宽度**（375px 屏上是 64px），
 *      而横向空间在手机上是最稀缺的资源
 *
 * 移动端惯例是**底部标签栏**（拇指自然可达）+ 抽屉承载剩余项。
 *
 * ★ 为什么底部只放 4 个 + "更多"
 * ---------------------------
 * 底部标签栏超过 5 个就会拥挤到点不准（每个目标 <60px 宽）。
 * 这里按使用频率挑 4 个日常高频页，其余 9 个放进"更多"抽屉——
 * 抽屉里是**带文字的完整列表**，可滚动、可搜索（当前项高亮）。
 */

/** 放进底部标签栏的高频页（按日常使用频率挑，非数组顺序）。 */
const PRIMARY_PATHS = ['/', '/chat', '/extract', '/price'] as const

const primaryItems = NAV_ITEMS.filter((item) =>
  (PRIMARY_PATHS as readonly string[]).includes(item.path),
)
const moreItems = NAV_ITEMS.filter(
  (item) => !(PRIMARY_PATHS as readonly string[]).includes(item.path),
)

export function MobileBottomNav({ onOpenMore }: { onOpenMore: () => void }) {
  const { pathname } = useLocation()

  /** 当前路由是否属于"更多"里的某一项（用于高亮"更多"按钮）。 */
  const inMore = moreItems.some((item) =>
    item.path === '/' ? pathname === '/' : pathname.startsWith(item.path),
  )

  return (
    <nav
      // pb-safe：iPhone 的 Home 指示条会盖住底部，不留安全区则最后一行点不到
      className="pb-safe fixed inset-x-0 bottom-0 z-30 border-t border-slate-200 bg-white/95 backdrop-blur lg:hidden"
      aria-label="主导航"
    >
      <div className="flex items-stretch">
        {primaryItems.map(({ path, label, icon: Icon }) => (
          <NavLink
            key={path}
            to={path}
            end={path === '/'}
            // touch-target：保证 ≥44px 可点区域；没有它图标只有 20px 高，很容易点空
            className={({ isActive }) =>
              `touch-target flex flex-1 flex-col items-center justify-center gap-0.5 py-2 text-[10px] transition-colors ${
                isActive ? 'text-brand-700' : 'text-ink-500'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <Icon className="h-5 w-5" strokeWidth={isActive ? 2.4 : 2} />
                <span className={isActive ? 'font-medium' : ''}>{label}</span>
              </>
            )}
          </NavLink>
        ))}

        {/* "更多"开抽屉。状态由 Layout 持有，这里只发回调——
            不用 CustomEvent 之类的跨组件广播：那会引入一条隐式依赖，
            读代码时看不出谁在监听（本项目其他位置也没有这种用法）。 */}
        <button
          type="button"
          onClick={onOpenMore}
          aria-label="打开全部功能菜单"
          className={`touch-target flex flex-1 flex-col items-center justify-center gap-0.5 py-2 text-[10px] transition-colors ${
            inMore ? 'text-brand-700' : 'text-ink-500'
          }`}
        >
          <Menu className="h-5 w-5" strokeWidth={inMore ? 2.4 : 2} />
          <span className={inMore ? 'font-medium' : ''}>更多</span>
        </button>
      </div>
    </nav>
  )
}

export function MobileNavDrawer({
  open,
  onClose,
}: {
  open: boolean
  onClose: () => void
}) {
  const { pathname } = useLocation()

  // 路由变化后自动关闭抽屉。
  //
  // 不关的话用户点了某一项、页面换了，抽屉仍盖在上面，
  // 看起来像"点了没反应"——移动端最常见的体验缺陷之一。
  useEffect(() => {
    onClose()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname])

  // 抽屉打开时锁定 body 滚动。
  //
  // 不锁的话背景页面会跟着手指滚动，而抽屉本身也在滚——
  // 用户会看到两层内容同时移动，很难操控。
  useEffect(() => {
    if (!open) return
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = prev
    }
  }, [open])

  // Esc 关闭（外接键盘的平板/桌面窄窗口）
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true">
      {/* 遮罩：点击关闭。bg-black/40 足够看出"下面还有内容"但不至于看不清 */}
      <button
        type="button"
        aria-label="关闭菜单"
        onClick={onClose}
        className="absolute inset-0 bg-black/40"
      />
      {/* 抽屉本体：从左侧滑入，宽度取 min(80vw, 320px)——
          太窄放不下中文标签，太宽在小屏上会占满整屏、失去"抽屉"的空间感 */}
      <div className="absolute inset-y-0 left-0 flex w-[min(80vw,320px)] flex-col bg-white shadow-xl">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
          <span className="text-[15px] font-bold text-ink-900">全部功能</span>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭"
            className="touch-target flex items-center justify-center rounded-lg text-ink-500 hover:bg-ink-50"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto px-3 py-3" aria-label="全部功能">
          {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              end={path === '/'}
              className={({ isActive }) =>
                `touch-target flex items-center gap-3 rounded-lg px-3 py-3 text-[14px] font-medium transition-colors ${
                  isActive
                    ? 'bg-brand-50 text-brand-700'
                    : 'text-ink-700 hover:bg-ink-50'
                }`
              }
            >
              <Icon className="h-5 w-5 shrink-0" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
      </div>
    </div>
  )
}
