import { useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { Sparkles } from 'lucide-react'
import { NAV_ITEMS } from '../config/nav'
import { MODEL_OPTIONS, useModel } from '../contexts/model'

export default function Layout() {
  const { model, setModel } = useModel()
  const location = useLocation()
  const [demoMode, setDemoMode] = useState(false)

  return (
    <div className="min-h-screen">
      {/* 左侧固定侧边栏: <1024px 收起为 64px 图标栏 */}
      <aside className="fixed inset-y-0 left-0 z-10 flex w-16 flex-col border-r border-slate-200 bg-white lg:w-[232px]">
        {/* Logo 区 */}
        <div className="flex items-center gap-3 border-b border-slate-200 px-3 py-4 lg:px-5">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-brand-500 to-brand-700">
            <Sparkles className="h-4 w-4 text-white" />
          </div>
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
                `relative flex items-center justify-center gap-3 rounded-lg px-3 py-2.5 text-[13px] font-medium transition-colors lg:justify-start ${
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
                  <span className="hidden lg:inline">{label}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* 演示模式开关: 切换全站等宽数字字体 */}
        <div className="hidden border-t border-slate-200 px-4 py-3 lg:block">
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
        <div className="hidden border-t border-slate-200 px-4 py-4 lg:block">
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

      {/* 右侧内容区: 页面切换 150ms 淡入, 演示模式下全站等宽数字 */}
      <main className="ml-16 min-h-screen p-8 lg:ml-[232px]">
        <div className={`mx-auto max-w-[1440px] ${demoMode ? 'demo-tabular' : ''}`}>
          <div key={location.pathname} className="page-fade">
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  )
}
