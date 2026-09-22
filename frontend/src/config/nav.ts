import {
  LayoutDashboard,
  MessageSquare,
  FileSearch,
  ShoppingCart,
  Package,
  BarChart3,
} from 'lucide-react'

export interface NavItem {
  path: string
  label: string
  icon: typeof LayoutDashboard
}

export const NAV_ITEMS: NavItem[] = [
  { path: '/', label: '工作台', icon: LayoutDashboard },
  { path: '/chat', label: '智能问答', icon: MessageSquare },
  { path: '/extract', label: '标讯智能提取', icon: FileSearch },
  { path: '/price', label: '智能商品比价', icon: ShoppingCart },
  { path: '/material', label: '物料智能识别', icon: Package },
  { path: '/analysis', label: '集采分析与推荐', icon: BarChart3 },
]
