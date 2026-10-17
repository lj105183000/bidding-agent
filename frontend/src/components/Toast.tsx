/* eslint-disable react-refresh/only-export-components -- 按设计规格: Provider 与 hook 同文件 */
import {
  createContext,
  useCallback,
  useContext,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { AlertTriangle, CheckCircle2, Info, XCircle } from 'lucide-react'

export type ToastType = 'success' | 'error' | 'warning' | 'info'

interface ToastItem {
  id: number
  message: string
  type: ToastType
  /** 入队时间戳，用于"同文案去重"的窗口判断 */
  at: number
}

interface ToastContextValue {
  /** 弹出提示, 2.5s 后自动消失 */
  show: (message: string, type?: ToastType) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast 必须在 <ToastProvider> 内使用')
  return ctx
}

const ICONS: Record<ToastType, ReactNode> = {
  success: <CheckCircle2 className="h-4 w-4 shrink-0 text-ok" />,
  error: <XCircle className="h-4 w-4 shrink-0 text-danger" />,
  warning: <AlertTriangle className="h-4 w-4 shrink-0 text-warn" />,
  info: <Info className="h-4 w-4 shrink-0 text-brand-600" />,
}

/** 同时存在的提示上限。
 *
 * ★ 为什么必须设上限
 * ----------------
 * 实测：后端未启动时，一个页面会并发发起十几个接口请求，
 * 每个失败都弹一次"网络异常，请检查后端服务" —— **屏幕上堆了十几张同文案的卡片**，
 * 把页面内容完全盖住（移动端截图实测，桌面端同样难看）。
 *
 * 提示过载比不提示更糟：用户只想知道"后端挂了"，不需要看十几遍。 */
const MAX_TOASTS = 3

/** 同文案去重窗口（毫秒）。
 *
 * 并发失败在 1 秒内会集中爆发，用 2000ms 足以把它们并成一条。
 * 窗口太短起不到合并作用，太长会让"用户重试后真的又失败"被静默吞掉。 */
const DEDUPE_WINDOW_MS = 2000

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([])
  const idRef = useRef(0)

  const show = useCallback((message: string, type: ToastType = 'info') => {
    const now = Date.now()
    const id = ++idRef.current
    setToasts((prev) => {
      // 同 (文案, 类型) 且在窗口内 → 视为重复，不重复弹。
      // 用 message+type 作键而不是只比 message：
      // 同一个文案在不同语义等级下（如"已保存"成功 vs 警告）应该分别显示。
      const dup = prev.some(
        (t) => t.message === message && t.type === type && now - t.at < DEDUPE_WINDOW_MS,
      )
      if (dup) return prev
      // 超出上限时丢掉**最旧的**：新提示通常与当前操作更相关
      const next = [...prev, { id, message, type, at: now }]
      return next.length > MAX_TOASTS ? next.slice(next.length - MAX_TOASTS) : next
    })
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id))
    }, 2500)
  }, [])

  return (
    <ToastContext.Provider value={{ show }}>
      {children}
      {/* 提示区位置与宽度
          
          ★ 移动端必须避开底部导航（实测缺陷）
          ------------------------------------
          原来是 `fixed bottom-6 right-6 w-80`：
            · 底部导航是 fixed 且高度约 56px（+ 安全区），bottom-6(24px) 的提示
              **整块被导航盖住** —— 用户在手机上完全看不到任何提示
            · w-80 = 320px，在 375px 屏上几乎占满整宽，且贴边 24px 后
              与屏幕右缘只剩 31px，视觉上很挤
          
          故窄屏改为**顶部居中**（移动端惯例，也是唯一不被底部导航与拇指
          遮挡的位置），宽屏保持右下角。`top` 要避开 sticky 顶栏（约 52px + 安全区）。 */}
      <div
        className="pointer-events-none fixed inset-x-3 top-[calc(3.5rem+env(safe-area-inset-top,0px))] z-50 flex flex-col gap-2 sm:inset-x-auto sm:bottom-6 sm:right-6 sm:top-auto sm:w-80"
        role="status"
        aria-live="polite"
      >
        {toasts.map((t) => (
          <div
            key={t.id}
            className="pointer-events-auto flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-3 shadow-lg"
          >
            {ICONS[t.type]}
            <span className="text-[13px] leading-5 text-ink-700">{t.message}</span>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}
