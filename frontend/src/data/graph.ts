/**
 * 知识图谱查询类型的展示元数据。
 *
 * 为什么单独成文件：查询类型的「契约」在后端（`rag_tools._GRAPH_QUERY_TYPES`），
 * 而「怎么显示」是纯前端的事——中文名、图标、示例问题、结果列定义放在这里，
 * 页面组件只负责渲染。后端新增类型时，这里补一条即可；
 * 清单本身通过 `fetchGraphTypes()` 从后端拉取，所以不会出现"前端有、后端没有"的幻觉条目。
 */
import {
  Boxes,
  GitBranch,
  Network,
  Search,
  TrendingUp,
  Truck,
} from 'lucide-react'

/** 结果列定义：把后端返回的原始字段映射成表头 + 渲染方式 */
export interface ColumnDef {
  key: string
  label: string
  /** number = 右对齐并用千分位；text = 普通文本；badge = 小徽章；mono = 等宽字体 */
  type?: 'text' | 'number' | 'badge' | 'mono'
}

export interface QueryTypeMeta {
  query_type: string
  label: string
  description: string
  icon: typeof Search
  /** 是否必须填实体名（与后端 /api/graph/types 的 requires_entity 一致） */
  requiresEntity: boolean
  /** 实体输入框的占位提示 */
  entityPlaceholder?: string
  /** 示例问题（点击直接填入） */
  examples: string[]
  /**
   * 结果表格的列定义。
   * - `graph_stats` / `entity_detail` 返回对象，不用表格，故此字段省略
   * - 字段名必须与 Neo4jClient 的返回键一致（见 src/database/neo4j_client.py）
   */
  columns?: ColumnDef[]
  /** 结果区标题（表格上方） */
  resultTitle?: string
}

export const QUERY_TYPES: QueryTypeMeta[] = [
  {
    query_type: 'search_entity',
    label: '实体检索',
    description: '按名称模糊搜索标的物 / 采购人 / 供应商',
    icon: Search,
    requiresEntity: true,
    entityPlaceholder: '输入实体名称，如：空调',
    examples: ['有哪些关于空调的采购？', '合肥市财政局', '安徽建工'],
    columns: [
      { key: 'name', label: '实体名称' },
      { key: 'label', label: '类型', type: 'badge' },
      { key: 'tradeFrequency', label: '交易频次', type: 'number' },
    ],
    resultTitle: '匹配到的实体',
  },
  {
    query_type: 'items_by_entity',
    label: '关联标的物',
    description: '查某个采购单位 / 供应商关联过哪些标的物',
    icon: GitBranch,
    requiresEntity: true,
    entityPlaceholder: '输入单位名称，如：合肥市财政局',
    examples: ['合肥市采购了什么？', '长丰县教育体育局', '安徽医科大学附属阜阳医院'],
    columns: [
      { key: 'subject', label: '标的物' },
      { key: 'relation', label: '关联方式', type: 'badge' },
      { key: 'entity', label: '关联实体' },
    ],
    resultTitle: '关联到的标的物',
  },
  {
    query_type: 'top_traded',
    label: '频次排行',
    description: '按采购交易频次排序的标的物榜',
    icon: TrendingUp,
    requiresEntity: false,
    examples: ['采购最多的标的物是什么？', '交易频次最高的品类'],
    columns: [
      { key: 'name', label: '标的物' },
      { key: 'tradeFrequency', label: '交易频次', type: 'number' },
    ],
    resultTitle: '采购频次排行',
  },
  {
    query_type: 'supplier_by_subject',
    label: '标的物供应商',
    description: '查某标的物的供应商及其最高中标金额',
    icon: Truck,
    requiresEntity: true,
    entityPlaceholder: '输入标的物，如：空调',
    examples: ['空调的供应商有哪些？', '复印纸的供应商'],
    columns: [
      { key: 'supplier', label: '供应商' },
      { key: 'amount', label: '最高中标金额（元）', type: 'number' },
    ],
    resultTitle: '该标的物的供应商',
  },
  {
    query_type: 'entity_detail',
    label: '完整关系',
    description: '某标的物的供应商 / 采购单位 / 代理机构 / 所在地 / 交易记录全景',
    icon: Network,
    requiresEntity: true,
    entityPlaceholder: '输入标的物，如：空调',
    examples: ['空调的供应商和采购人有哪些？', '粉盒的完整关系'],
    resultTitle: '标的物关联信息',
  },
  {
    query_type: 'graph_stats',
    label: '图谱统计',
    description: '各类型节点与关系的规模总览',
    icon: Boxes,
    requiresEntity: false,
    examples: ['知识图谱中有多少数据？'],
    resultTitle: '图谱规模',
  },
]

/** 按 query_type 取元数据；未登记的类型返回一个保守的兜底（不崩、不瞎编列） */
export function getQueryTypeMeta(queryType: string): QueryTypeMeta | undefined {
  return QUERY_TYPES.find((t) => t.query_type === queryType)
}

/** 节点标签的中文名（用于 graph_stats 的对象渲染与 badge 列） */
export const LABEL_NAMES: Record<string, string> = {
  SubjectMatter: '标的物',
  Purchaser: '采购单位',
  Supplier: '供应商',
  Agency: '代理机构',
  Location: '地区',
  Transaction: '交易记录',
}

/** 关系类型的中文名（items_by_entity 的 relation 列） */
export const RELATION_NAMES: Record<string, string> = {
  HAS_TRANSACTION: '交易记录',
  PURCHASED_BY: '采购单位',
  SUPPLIED_BY: '供应商',
  AGENT_BY: '代理机构',
  LOCATED_AT: '所在地',
}

/** entity_detail 的分段标题（键 → 中文），顺序即展示顺序 */
export const DETAIL_SECTIONS: { key: string; title: string }[] = [
  { key: 'suppliers', title: '供应商' },
  { key: 'purchasers', title: '采购单位' },
  { key: 'agencies', title: '代理机构' },
  { key: 'locations', title: '所在地' },
  { key: 'transactions', title: '交易记录' },
]

// ---------------------------------------------------------------------------
// 首页示例问题（一键直达 /graph 并自动执行）
// ---------------------------------------------------------------------------

/**
 * `/graph` 页通过 react-router 的 location.state 接收的载荷。
 *
 * 为什么用 state 而不是 query string：这些问题含中文与问号，
 * 拼进 URL 需要 encode 且会让地址栏变得嘈杂；state 是内存传递，
 * 刷新即失效——正合语义（"点一次示例就跑一次"）。
 */
export interface GraphNavPayload {
  /** 标记这是"从示例跳转而来"，页面据此决定是否自动执行 */
  autoRun?: boolean
  query_type?: string
  entity?: string
  /** 示例原文，用于结果区顶部回显"你问的是哪个问题" */
  question?: string
}

export interface GraphExample {
  /** 展示在首页的问题文案 */
  question: string
  /** 点击后跳转的目标路由 */
  path: '/graph' | '/chat'
  /** 图谱类示例的执行参数；跳问答页时不需要 */
  query_type?: string
  entity?: string
  /** 一句话说明这条示例在演示什么能力（帮助用户理解数据源差异） */
  source: string
}

/**
 * 首页「示例问题」区的数据。
 *
 * 这几条与 `QUERY_TYPES[].examples` 是**同一批问题**，但用途不同：
 *   - `QUERY_TYPES[].examples`：在 /graph 页内快速填充当前类型的输入框
 *   - 这里：在首页跨页直达某个能力，故需额外带 `path` 与 `query_type`
 * 为避免两处漂移，下方有单测断言"这里的每条形如 /graph 的示例都能匹配到已登记的类型"。
 */
export const HOME_EXAMPLES: GraphExample[] = [
  {
    question: '有哪些关于空调的采购？',
    path: '/graph',
    query_type: 'search_entity',
    entity: '空调',
    source: '知识图谱 · 实体检索',
  },
  {
    question: '采购最多的标的是什么？',
    path: '/graph',
    query_type: 'top_traded',
    source: '知识图谱 · 频次排行',
  },
  {
    question: '空调的供应商和采购人有哪些？',
    path: '/graph',
    query_type: 'entity_detail',
    entity: '空调',
    source: '知识图谱 · 完整关系',
  },
  {
    question: '中标金额最高的项目是哪个？',
    path: '/chat',
    source: '业务数据库 · 对话式查询',
  },
]

/** 由示例生成跳转用的 state；跳问答页时只让页面聚焦输入框，不自动提问 */
export function buildExamplePayload(example: GraphExample): GraphNavPayload | undefined {
  if (example.path !== '/graph') return undefined
  return {
    autoRun: true,
    query_type: example.query_type,
    entity: example.entity,
    question: example.question,
  }
}
