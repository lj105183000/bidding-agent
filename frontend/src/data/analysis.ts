// 集采分析与推荐 mock 数据
// TOP_SUPPLIERS 来源于 reference/bids.xlsx 真实中标人统计(按累计中标金额排序, 金额单位: 万元)
interface Kpi {
  totalBids: number
  provinces: number
  suppliers: number
  savingRate: number
}

export const KPI: Kpi = {
  totalBids: 12846,
  provinces: 28,
  suppliers: 3215,
  savingRate: 12.7,
}

export const CATEGORY_DIST = [
  { name: '货物', value: 42 },
  { name: '服务', value: 31 },
  { name: '工程', value: 27 },
]

/** 1-10 月集采金额(万元) */
export const MONTHLY_AMOUNT = [3200, 2850, 4100, 3980, 4560, 5200, 4880, 5420, 6100, 5860]

interface SupplierStat {
  name: string
  count: number
  /** 累计中标金额(万元) */
  amount: number
}

export const TOP_SUPPLIERS: SupplierStat[] = [
  { name: '合肥瑶海静安养亲护养院', count: 1, amount: 660000.0 },
  { name: '安徽建工三建集团有限公司', count: 1, amount: 236760.6 },
  { name: '安徽省公路桥梁工程有限公司', count: 2, amount: 133375.6 },
  { name: '中国建筑第四工程局有限公司', count: 1, amount: 110103.0 },
  { name: '中建六局第八建设有限公司', count: 1, amount: 99712.0 },
  { name: '南京上铁地方铁路开发有限公司', count: 1, amount: 98197.0 },
  { name: '深圳市建筑设计研究总院有限公司', count: 2, amount: 71927.3 },
  { name: '韩大建设有限公司', count: 1, amount: 54185.7 },
  { name: '安徽省城建设计研究总院股份有限公司', count: 1, amount: 48604.9 },
  { name: '中国建筑第八工程局有限公司', count: 1, amount: 47156.2 },
]

interface RecommendSupplier {
  id: string
  name: string
  matchScore: number
  cooperation: number
  rating: number
  category: string
  region: string
  tags: string[]
}

export const RECOMMEND_SUPPLIERS: RecommendSupplier[] = [
  {
    id: 'sup-001',
    name: '西域智慧供应链(上海)股份公司',
    matchScore: 98.5,
    cooperation: 326,
    rating: 4.9,
    category: '综合工业品',
    region: '上海',
    tags: ['MRO集采', '全国仓配', '48h交付'],
  },
  {
    id: 'sup-002',
    name: '震坤行工业超市(上海)有限公司',
    matchScore: 97.2,
    cooperation: 289,
    rating: 4.8,
    category: '综合工业品',
    region: '上海',
    tags: ['数字化采购', '品类齐全', '驻场服务'],
  },
  {
    id: 'sup-003',
    name: '鑫方盛数智科技股份有限公司',
    matchScore: 95.8,
    cooperation: 203,
    rating: 4.7,
    category: '五金机电',
    region: '北京',
    tags: ['五金工具', '价格优势', '区域仓配'],
  },
  {
    id: 'sup-004',
    name: '京东工业品',
    matchScore: 87.6,
    cooperation: 156,
    rating: 4.8,
    category: '综合工业品',
    region: '北京',
    tags: ['电商直营', '次日达', '大牌授权'],
  },
  {
    id: 'sup-005',
    name: '得力集实(杭州)科技有限公司',
    matchScore: 79.5,
    cooperation: 98,
    rating: 4.6,
    category: '劳保办公',
    region: '杭州',
    tags: ['劳保用品', '办公物资', '应急保障'],
  },
]
