/**
 * 图谱查询类型配置的最小单测（与 api.test.ts 同级，纯函数、不铺开测试框架）。
 *
 * 覆盖点：
 *   - 6 类查询类型与后端 _GRAPH_QUERY_TYPES 一一对应（防漂移）
 *   - requiresEntity 与后端 /api/graph/types 的契约一致
 *   - 需要实体的类型必须给出占位提示；结果列必须非空
 *   - 标签/关系的中文映射覆盖后端全部 ALLOWED_LABELS / ALLOWED_RELS
 */
import { describe, it, expect } from 'vitest'
import {
  DETAIL_SECTIONS,
  HOME_EXAMPLES,
  LABEL_NAMES,
  QUERY_TYPES,
  RELATION_NAMES,
  buildExamplePayload,
  getQueryTypeMeta,
} from './graph'

/** 后端 src/tools/rag_tools.py 的 _GRAPH_QUERY_TYPES（人工同步，改了这里就该改那边） */
const BACKEND_QUERY_TYPES = [
  'search_entity',
  'items_by_entity',
  'top_traded',
  'entity_detail',
  'supplier_by_subject',
  'graph_stats',
]

/** 后端 src/database/neo4j_client.py 的 ALLOWED_LABELS / ALLOWED_RELS */
const BACKEND_LABELS = [
  'SubjectMatter',
  'Purchaser',
  'Agency',
  'Location',
  'Supplier',
  'Transaction',
]
const BACKEND_RELS = [
  'HAS_TRANSACTION',
  'PURCHASED_BY',
  'AGENT_BY',
  'LOCATED_AT',
  'SUPPLIED_BY',
]

describe('QUERY_TYPES 与后端契约一致', () => {
  it('恰好覆盖后端全部查询类型，不多不少', () => {
    expect(QUERY_TYPES.map((t) => t.query_type).sort()).toEqual([...BACKEND_QUERY_TYPES].sort())
  })

  it('requiresEntity 与后端 /api/graph/types 的声明一致', () => {
    // 后端 app.py graph_types() 里的 needs_entity 映射
    const backendNeedsEntity: Record<string, boolean> = {
      search_entity: true,
      items_by_entity: true,
      entity_detail: true,
      supplier_by_subject: true,
      top_traded: false,
      graph_stats: false,
    }
    for (const t of QUERY_TYPES) {
      expect(t.requiresEntity, `${t.query_type} 的 requiresEntity`).toBe(
        backendNeedsEntity[t.query_type],
      )
    }
  })

  it('每类都有中文名、描述和至少一个示例问题', () => {
    for (const t of QUERY_TYPES) {
      expect(t.label.length, `${t.query_type} 缺少中文名`).toBeGreaterThan(0)
      expect(t.description.length, `${t.query_type} 缺少描述`).toBeGreaterThan(0)
      expect(t.examples.length, `${t.query_type} 缺少示例`).toBeGreaterThan(0)
    }
  })

  it('需要实体的类型必须给输入框占位提示', () => {
    for (const t of QUERY_TYPES.filter((x) => x.requiresEntity)) {
      expect(t.entityPlaceholder, `${t.query_type} 缺少 entityPlaceholder`).toBeTruthy()
    }
  })

  it('数组型结果（4 类）必须有非空列定义；对象型（2 类）不定义列', () => {
    const objectTypes = ['graph_stats', 'entity_detail']
    for (const t of QUERY_TYPES) {
      if (objectTypes.includes(t.query_type)) {
        expect(t.columns, `${t.query_type} 返回对象，不应定义表格列`).toBeUndefined()
      } else {
        expect(t.columns?.length, `${t.query_type} 缺少列定义`).toBeGreaterThan(0)
        for (const col of t.columns ?? []) {
          expect(col.key, `${t.query_type} 的列缺 key`).toBeTruthy()
          expect(col.label, `${t.query_type}.${col.key} 缺中文表头`).toBeTruthy()
        }
      }
    }
  })
})

describe('getQueryTypeMeta', () => {
  it('已知类型返回配置', () => {
    expect(getQueryTypeMeta('top_traded')?.label).toBe('频次排行')
  })

  it('未知类型返回 undefined（页面据此走兜底，不崩）', () => {
    expect(getQueryTypeMeta('not_a_real_type')).toBeUndefined()
  })
})

describe('中英映射覆盖后端白名单', () => {
  it('LABEL_NAMES 覆盖全部节点标签', () => {
    for (const label of BACKEND_LABELS) {
      expect(LABEL_NAMES[label], `缺少标签映射 ${label}`).toBeTruthy()
    }
  })

  it('RELATION_NAMES 覆盖全部关系类型', () => {
    for (const rel of BACKEND_RELS) {
      expect(RELATION_NAMES[rel], `缺少关系映射 ${rel}`).toBeTruthy()
    }
  })

  it('DETAIL_SECTIONS 每个键都有中文标题', () => {
    expect(DETAIL_SECTIONS.length).toBeGreaterThan(0)
    for (const s of DETAIL_SECTIONS) {
      expect(s.key).toBeTruthy()
      expect(s.title).toBeTruthy()
    }
  })
})

describe('首页示例问题', () => {
  it('三条图谱示例都在（用户明确要求加入的）', () => {
    const questions = HOME_EXAMPLES.map((e) => e.question)
    expect(questions).toContain('有哪些关于空调的采购？')
    expect(questions).toContain('采购最多的标的是什么？')
    expect(questions).toContain('空调的供应商和采购人有哪些？')
  })

  it('每条都写了问题文案与来源说明', () => {
    for (const ex of HOME_EXAMPLES) {
      expect(ex.question.length, '示例缺少问题文案').toBeGreaterThan(0)
      expect(ex.source.length, `${ex.question} 缺少来源说明`).toBeGreaterThan(0)
      expect(['/graph', '/chat'], `${ex.question} 的目标路由非法`).toContain(ex.path)
    }
  })

  it('跳图谱的示例必须指向已登记的查询类型，否则会打开就报 400', () => {
    const known = new Set(QUERY_TYPES.map((t) => t.query_type))
    for (const ex of HOME_EXAMPLES.filter((e) => e.path === '/graph')) {
      expect(ex.query_type, `${ex.question} 未指定 query_type`).toBeTruthy()
      expect(known.has(ex.query_type!), `${ex.question} 指向未登记类型 ${ex.query_type}`).toBe(true)
    }
  })

  it('需要实体的示例必须带 entity，否则跳过去只会弹一条警告', () => {
    for (const ex of HOME_EXAMPLES.filter((e) => e.path === '/graph')) {
      const meta = getQueryTypeMeta(ex.query_type!)
      if (meta?.requiresEntity) {
        expect(ex.entity, `${ex.question} 的类型需要实体但没给 entity`).toBeTruthy()
      }
    }
  })

  it('图谱示例的 payload 带 autoRun 与回显用的问题原文', () => {
    const graphEx = HOME_EXAMPLES.filter((e) => e.path === '/graph')
    expect(graphEx.length).toBeGreaterThan(0)
    for (const ex of graphEx) {
      const payload = buildExamplePayload(ex)
      expect(payload?.autoRun).toBe(true)
      expect(payload?.question).toBe(ex.question)
      expect(payload?.query_type).toBe(ex.query_type)
    }
  })

  it('跳问答页的示例不产生图谱 payload（以免被误当成自动执行）', () => {
    for (const ex of HOME_EXAMPLES.filter((e) => e.path === '/chat')) {
      expect(buildExamplePayload(ex)).toBeUndefined()
    }
  })
})
