import {
  LayoutDashboard,
  MessageSquare,
  Network,
  FileSearch,
  ShoppingCart,
  Package,
  BarChart3,
  Share2,
  FileText,
  FileDiff,
  BadgeCheck,
  Scale,
  CalendarClock,
  ClipboardCheck,
} from 'lucide-react'

export interface NavItem {
  path: string
  label: string
  icon: typeof LayoutDashboard
}

export const NAV_ITEMS: NavItem[] = [
  { path: '/', label: '工作台', icon: LayoutDashboard },
  { path: '/chat', label: '智能问答', icon: MessageSquare },
  // 知识图谱：显式指定查询类型直查 Neo4j（与问答页的"Agent 自己决定"互补）
  { path: '/graph', label: '知识图谱', icon: Network },
  // 图谱可视化：力导向图展示关系网络（与上一条的表格视图互补）
  { path: '/graph-viz', label: '图谱可视化', icon: Share2 },
  { path: '/extract', label: '标讯智能提取', icon: FileSearch },
  // 招标文件解析：面向 PDF/Word 正文（与上一条的"公告文本/截图"互补）
  { path: '/tender', label: '招标文件解析', icon: FileText },
  // 变更追踪：原文件 vs 澄清文件，防漏看变更导致废标
  { path: '/compare', label: '变更追踪', icon: FileDiff },
  // 资格审查：录入企业档案，逐条比对资格要求
  { path: '/qualification', label: '资格审查', icon: BadgeCheck },
  // 合规检查：规则检出 + 法规引用（结论为「疑似」而非违规认定）
  { path: '/compliance', label: '合规性检查', icon: Scale },
  // 截止监控：只覆盖已上传并抽到截止时间的文件（不是全网监控）
  { path: '/deadline', label: '截止日期监控', icon: CalendarClock },
  { path: '/price', label: '智能商品比价', icon: ShoppingCart },
  { path: '/material', label: '物料智能识别', icon: Package },
  { path: '/analysis', label: '集采分析与推荐', icon: BarChart3 },
  // 评估看板：离线评测（固定测试集 + LLM 裁判）与在线评测（真实请求采样）
  //
  // ★ 为什么放在最后
  //   它是**开发者/运维视角**的页面（指标有没有退化、检索准不准），
  //   而上面那些是业务用户日常要用的。放最后符合实际使用频率。
  { path: '/eval', label: '评估看板', icon: ClipboardCheck },
]
