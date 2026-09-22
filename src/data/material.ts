// 物料智能识别 mock 数据: 一物多码场景, 3 组物料各含 3 条来自 ERP/MES/SRM 的原始描述
interface RawMaterialDesc {
  source: string
  text: string
}

export interface MaterialGroup {
  id: string
  name: string
  stdCode: string
  categoryPath: string
  spec: string
  brand: string
  matchScore: number
  rawDescriptions: RawMaterialDesc[]
}

export const MATERIAL_GROUPS: MaterialGroup[] = [
  {
    id: 'mat-001',
    name: '球头内六角扳手',
    stdCode: 'GY-BS-000117',
    categoryPath: '五金工具 > 手动工具 > 扳手类 > 内六角扳手',
    spec: '英制 1/16"-3/8"，加长球头，9件套',
    brand: '波斯 BOSI',
    matchScore: 98.6,
    rawDescriptions: [
      { source: 'ERP', text: '内六角扳手 9件套 英制 加长球头 1/16-3/8' },
      { source: 'MES', text: '波斯 BS423181 球头内六角扳手组套 英制九件装' },
      { source: 'SRM', text: '六角扳手 英制 球头加长 套装 9PCS BOSI' },
    ],
  },
  {
    id: 'mat-002',
    name: '微型断路器',
    stdCode: 'GY-DQ-002341',
    categoryPath: '电工电气 > 低压电器 > 断路器 > 微型断路器',
    spec: '2P C型 32A 6kA',
    brand: '施耐德',
    matchScore: 96.2,
    rawDescriptions: [
      { source: 'ERP', text: '断路器 2P C32A 施耐德' },
      { source: 'MES', text: '施耐德 iC65N 2P C32 微型断路器 6kA' },
      { source: 'SRM', text: '微断 2P 32A C曲线 iC65N 施耐德电气' },
    ],
  },
  {
    id: 'mat-003',
    name: 'KN95 防护口罩',
    stdCode: 'GY-LB-000892',
    categoryPath: '劳保用品 > 呼吸防护 > 防尘口罩 > KN95口罩',
    spec: '头戴式 KN95 50只/盒',
    brand: '3M',
    matchScore: 94.8,
    rawDescriptions: [
      { source: 'ERP', text: '3M 9502+ 防尘口罩 KN95 头戴式' },
      { source: 'MES', text: 'KN95 口罩 头戴 50只/盒 3M' },
      { source: 'SRM', text: '3M口罩 9502+ KN95 头戴式 50只装' },
    ],
  },
]
