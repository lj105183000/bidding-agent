// 标讯智能提取演示样例: rawText 为基于 reference/bids.xlsx 真实记录字段还原的公告原文
// expected 为对应记录的结构化字段(17 字段中的 14 个关键字段)
import type { BidRecord } from './bids'

export type ExpectedFields = Omit<BidRecord, 'id'>

interface ExtractSample {
  id: string
  name: string
  rawText: string
  expected: ExpectedFields
}

export const EXTRACT_SAMPLES: ExtractSample[] = [
  {
    id: 'ex-001',
    name: '合肥合燃华润燃气有限公司超声波流量计检测服务项目中标结果公告',
    rawText:
      '合肥合燃华润燃气有限公司超声波流量计检测服务项目中标结果公告\n' +
      '项目编号：2023BFFWZ01956。经评标委员会评审，合肥合燃华润燃气有限公司超声波流量计检测服务项目确定中标人为重庆稳信科技有限公司，中标金额366500元。\n' +
      '本项目为服务类招标采购项目，招标人为合肥合燃华润燃气有限公司。公告时间：2023年10月13日。',
    expected: {
      title: '合肥合燃华润燃气有限公司超声波流量计检测服务项目',
      category: '服务',
      source: '招标采购',
      publishTime: '2023-10-13 00:00:00',
      province: '安徽省',
      city: '合肥市',
      county: '',
      projectNo: '2023BFFWZ01956',
      projectName: '合肥合燃华润燃气有限公司超声波流量计检测服务项目',
      purchaser: '合肥合燃华润燃气有限公司',
      agency: '',
      budget: '',
      address: '合肥合燃华润燃气有限公司',
      period: '',
      winner: '重庆稳信科技有限公司',
      winAmount: 366500,
      winTime: '2023-10-13 18:17:16',
    },
  },
  {
    id: 'ex-002',
    name: '庐江县2023年大气污染防控精准溯源服务项目中标结果公告',
    rawText:
      '庐江县2023年大气污染防控精准溯源服务项目中标结果公告\n' +
      '项目编号：2023ANNFZ00411。庐江县2023年大气污染防控精准溯源服务项目属政府采购项目，采购人为合肥市庐江县生态环境分局，代理机构为安徽忠之盈工程咨询有限公司。\n' +
      '经评审确定中标人为安徽科创中光科技股份有限公司，中标金额1684000元。项目地点位于合肥市庐江县，公告时间：2023年10月13日。',
    expected: {
      title: '庐江县2023年大气污染防控精准溯源服务项目见证书',
      category: '服务',
      source: '政府采购',
      publishTime: '2023-10-13 00:00:00',
      province: '安徽省',
      city: '合肥市',
      county: '庐江县',
      projectNo: '2023ANNFZ00411',
      projectName: '庐江县2023年大气污染防控精准溯源服务项目',
      purchaser: '合肥市庐江县生态环境分局',
      agency: '安徽忠之盈工程咨询有限公司',
      budget: '',
      address: '合肥市庐江县生态环境分局',
      period: '',
      winner: '安徽科创中光科技股份有限公司',
      winAmount: 1684000,
      winTime: '2023-10-13 18:18:30.213000',
    },
  },
  {
    id: 'ex-003',
    name: '2023年度北环高速路面专项养护工程中标通知书',
    rawText:
      '2023年度北环高速路面专项养护工程中标通知书\n' +
      '项目编号：2023BFFGZ01925。\n' +
      '安徽国路高速公路有限公司：\n' +
      '经评标委员会评定，2023年度北环高速路面专项养护工程确定中标人为安徽交控工程集团有限公司，中标金额12232306.89元。\n' +
      '本项目由安徽公共资源交易集团项目管理有限公司代理招标，属于工程类招标采购项目，请中标人在接到本通知书后30日内与招标人签订合同。\n' +
      '2023年10月16日',
    expected: {
      title: '2023年度北环高速路面专项养护工程中标通知书',
      category: '工程',
      source: '招标采购',
      publishTime: '2023-10-16 00:00:00',
      province: '安徽省',
      city: '合肥市',
      county: '',
      projectNo: '2023BFFGZ01925',
      projectName: '2023年度北环高速路面专项养护工程',
      purchaser: '安徽国路高速公路有限公司',
      agency: '安徽公共资源交易集团项目管理有限公司',
      budget: '',
      address: '安徽国路高速公路有限公司',
      period: '',
      winner: '安徽交控工程集团有限公司',
      winAmount: 12232306.89,
      winTime: '2023-10-16 13:44:51.147000',
    },
  },
]
