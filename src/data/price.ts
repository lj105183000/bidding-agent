// 智能商品比价 mock 数据: 参考价与多平台报价, 含近 6 个月价格走势(2026-03 ~ 2026-08)
export interface PriceItem {
  id: string
  name: string
  spec: string
  unit: string
  brand: string
  refPrice: number
  platforms: { name: string; price: number }[]
  trend: { month: string; price: number }[]
}

export const PRICE_DATA: PriceItem[] = [
  {
    id: 'price-001',
    name: '波斯 加长球头内六角扳手',
    spec: 'BS423181 1/16"-3/8" 1套',
    unit: '套',
    brand: '波斯 BOSI',
    refPrice: 26.73,
    platforms: [
      { name: '西域', price: 28.39 },
      { name: '京东', price: 26.53 },
      { name: '震坤行', price: 27.34 },
      { name: '鑫方盛', price: 26.89 },
    ],
    trend: [
      { month: '2026-03', price: 28.4 },
      { month: '2026-04', price: 28.2 },
      { month: '2026-05', price: 27.9 },
      { month: '2026-06', price: 27.5 },
      { month: '2026-07', price: 27.1 },
      { month: '2026-08', price: 26.73 },
    ],
  },
  {
    id: 'price-002',
    name: '施耐德 微型断路器',
    spec: 'iC65N 2P C32A',
    unit: '只',
    brand: '施耐德',
    refPrice: 78.5,
    platforms: [
      { name: '西域', price: 79.9 },
      { name: '京东', price: 77.2 },
      { name: '震坤行', price: 80.15 },
      { name: '鑫方盛', price: 78.3 },
    ],
    trend: [
      { month: '2026-03', price: 81.2 },
      { month: '2026-04', price: 80.5 },
      { month: '2026-05', price: 79.8 },
      { month: '2026-06', price: 79.0 },
      { month: '2026-07', price: 78.6 },
      { month: '2026-08', price: 78.5 },
    ],
  },
  {
    id: 'price-003',
    name: '3M 防护口罩 9502+',
    spec: 'KN95 头戴式 50只/盒',
    unit: '盒',
    brand: '3M',
    refPrice: 198.0,
    platforms: [
      { name: '西域', price: 205.0 },
      { name: '京东', price: 195.5 },
      { name: '震坤行', price: 199.9 },
      { name: '鑫方盛', price: 201.2 },
    ],
    trend: [
      { month: '2026-03', price: 212.0 },
      { month: '2026-04', price: 208.5 },
      { month: '2026-05', price: 205.0 },
      { month: '2026-06', price: 201.5 },
      { month: '2026-07', price: 199.0 },
      { month: '2026-08', price: 198.0 },
    ],
  },
  {
    id: 'price-004',
    name: '西门子 交流接触器',
    spec: '3TF30 22-0X 220V',
    unit: '台',
    brand: '西门子',
    refPrice: 156.8,
    platforms: [
      { name: '西域', price: 158.9 },
      { name: '京东', price: 154.0 },
      { name: '震坤行', price: 159.6 },
      { name: '鑫方盛', price: 157.45 },
    ],
    trend: [
      { month: '2026-03', price: 162.5 },
      { month: '2026-04', price: 161.0 },
      { month: '2026-05', price: 159.5 },
      { month: '2026-06', price: 158.2 },
      { month: '2026-07', price: 157.3 },
      { month: '2026-08', price: 156.8 },
    ],
  },
]
