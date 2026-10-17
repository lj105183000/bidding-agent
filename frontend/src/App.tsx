import { lazy } from 'react'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import Layout from './components/Layout'
import { ToastProvider } from './components/Toast'
import { ModelProvider } from './contexts/model'
import { NotFound, RouteError } from './components/RouteFallback'

/**
 * 页面组件一律**懒加载**（路由级代码分割）。
 *
 * ★ 为什么必须拆（实测数据，不是"优化习惯"）
 * ----------------------------------------
 * 拆之前 14 个页面全是静态 import，构建产物只有 3 个 chunk，且**全部随首屏下载**：
 *
 *     index   451.1 kB   全部 14 个页面 + react-markdown + 图谱渲染
 *     charts  359.1 kB   recharts —— 但只有 4 个页面用得到
 *     react   201.2 kB
 *     ────────────────
 *     首屏 1011.4 kB（gzip ≈303 kB）
 *
 * 也就是说：只想看一眼工作台的用户，要先下完智能问答（单页 201.8 kB）、
 * 图表库（359.1 kB）与另外 12 个页面。改成懒加载后实测首屏只剩
 * **242.0 kB**（entry 40.8 + react 201.2）—— **−769 kB / −76%**，
 * 其余按路由按需加载（`Analysis` 26.2、`Chat` 201.8、`Dashboard` 9.4 …）。
 *
 * ★ Suspense 挂在 Layout 里，不在这里
 * ---------------------------------
 * `Layout` 的 `<Outlet/>` 外面包了一层 `Suspense`（见 `components/Layout.tsx`）：
 * 这样等待页面 chunk 时**侧栏与移动端顶栏仍然在**，用户可以直接切走，
 * 而不是面对一个整屏占位。故这里只负责把组件换成 `lazy(...)`。
 *
 * ★ 为什么不用 `import()` 的注释魔法做"手动分包"
 * ------------------------------------------
 * 路由边界就是天然的切分点：它同时是**用户意图的边界**（点哪页下哪页）
 * 与**代码的边界**（页面之间几乎不互相 import）。按 vendor 名字硬拆
 * （如再拆一个 markdown chunk）反而不如路由粒度直观，且 Vite 已自动把
 * 被多个懒 chunk 共用的模块提成共享 chunk（实测 `api` 14.2 kB、`graph` 5.4 kB）。
 */
const Dashboard = lazy(() => import('./pages/Dashboard'))
const Chat = lazy(() => import('./pages/Chat'))
const KnowledgeGraph = lazy(() => import('./pages/KnowledgeGraph'))
const GraphViz = lazy(() => import('./pages/GraphViz'))
const Extract = lazy(() => import('./pages/Extract'))
const TenderDoc = lazy(() => import('./pages/TenderDoc'))
const Compare = lazy(() => import('./pages/Compare'))
const Qualification = lazy(() => import('./pages/Qualification'))
const Compliance = lazy(() => import('./pages/Compliance'))
const Deadline = lazy(() => import('./pages/Deadline'))
const PriceCompare = lazy(() => import('./pages/PriceCompare'))
const Material = lazy(() => import('./pages/Material'))
const Analysis = lazy(() => import('./pages/Analysis'))
const EvalDashboard = lazy(() => import('./pages/EvalDashboard'))

/**
 * 路由表。
 *
 * `errorElement` 挂在**根路由**上（而不是每个子路由各写一份）：
 * 这样无论是"路径没匹配"还是"某个页面渲染抛错"，都会落到我们自己的兜底页，
 * 而不是 React Router 自带的开发者错误页（"Unexpected Application Error! …
 * 💿 Hey developer 👋"）——那个页面不告诉用户错在哪，还暴露技术细节。
 *
 * `path: '*'` 是**显式的兜底子路由**：`errorElement` 只在"抛错/无匹配"时
 * 由 Router 内部触发，加一条 catch-all 能让未匹配路径也走正常的 Layout 渲染路径
 * （保留侧边导航，用户可以直接点去别的页面，而不是面对一个孤立页）。
 */
const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'chat', element: <Chat /> },
      { path: 'graph', element: <KnowledgeGraph /> },
      { path: 'graph-viz', element: <GraphViz /> },
      { path: 'extract', element: <Extract /> },
      { path: 'tender', element: <TenderDoc /> },
      { path: 'compare', element: <Compare /> },
      { path: 'qualification', element: <Qualification /> },
      { path: 'compliance', element: <Compliance /> },
      { path: 'deadline', element: <Deadline /> },
      { path: 'price', element: <PriceCompare /> },
      { path: 'material', element: <Material /> },
      { path: 'analysis', element: <Analysis /> },
      { path: 'eval', element: <EvalDashboard /> },
      // 兜底：未匹配的路径也留在 Layout 内，侧边导航仍可用
      { path: '*', element: <NotFound /> },
    ],
  },
])

export default function App() {
  return (
    <ModelProvider>
      <ToastProvider>
        <RouterProvider router={router} />
      </ToastProvider>
    </ModelProvider>
  )
}
