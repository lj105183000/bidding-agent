import { describe, expect, it } from 'vitest'
import {
  MAX_ATTACHMENTS,
  MAX_IMAGE_BYTES,
  describeAdditions,
  mergeImageAnalyses,
  planAdditions,
  splitByKind,
} from './imageQueue'

/**
 * 多图附件的三条规则。
 *
 * ★ 为什么这些用例重要：多图上传的错误都**不会报错**，只会悄悄出错 ——
 * 少挂一张、把两张图的结论串了、被拒了却不说原因。手动测通常只传一两张图，
 * 覆盖不到这些边界。
 */
const img = (size: number, type = 'image/png') => ({ type, size })

describe('planAdditions', () => {
  it('名额够时全部接收', () => {
    const plan = planAdditions(0, [img(1000), img(2000)])
    expect(plan.accepted).toEqual([0, 1])
    expect(plan.rejected).toEqual([])
  })

  it('★ 已在队列里的图占名额（先到先得）', () => {
    const plan = planAdditions(MAX_ATTACHMENTS - 1, [img(1000), img(1000)])
    expect(plan.accepted).toEqual([0])
    expect(plan.rejected).toEqual([{ index: 1, reason: 'too_many' }])
  })

  it('★ 名额已满时全部拒绝，且原因是 too_many', () => {
    const plan = planAdditions(MAX_ATTACHMENTS, [img(1000)])
    expect(plan.accepted).toEqual([])
    expect(plan.rejected[0].reason).toBe('too_many')
  })

  it('非图片文件被拒（拖拽常混入 pdf/zip）', () => {
    const plan = planAdditions(0, [img(1000, 'application/pdf'), img(1000, '')])
    expect(plan.accepted).toEqual([])
    expect(plan.rejected.map((r) => r.reason)).toEqual(['not_image', 'not_image'])
  })

  it('超过体积上限的图片被拒', () => {
    const plan = planAdditions(0, [img(MAX_IMAGE_BYTES + 1)])
    expect(plan.rejected).toEqual([{ index: 0, reason: 'too_large' }])
  })

  it('恰好等于上限时接收（边界不误伤）', () => {
    expect(planAdditions(0, [img(MAX_IMAGE_BYTES)]).accepted).toEqual([0])
  })

  it('类型与体积的判定优先于名额（原因要准确）', () => {
    // 名额已满，但第 0 个是 pdf —— 应报 not_image 而不是 too_many
    const plan = planAdditions(MAX_ATTACHMENTS, [img(10, 'application/pdf')])
    expect(plan.rejected[0].reason).toBe('not_image')
  })

  it('上限可覆盖（便于以后按部署调整）', () => {
    const plan = planAdditions(0, [img(10), img(10), img(10)], { maxCount: 2 })
    expect(plan.accepted).toEqual([0, 1])
    expect(plan.rejected[0].reason).toBe('too_many')
  })

  it('空输入安全', () => {
    expect(planAdditions(0, [])).toEqual({ accepted: [], rejected: [] })
  })
})

describe('splitByKind', () => {
  it('把图片、PDF、其它分开（顺序保持不变）', () => {
    const files = [
      { type: 'image/png', name: 'a.png' },
      { type: 'application/pdf', name: 'b.pdf' },
      { type: 'application/zip', name: 'c.zip' },
      { type: 'image/jpeg', name: 'd.jpg' },
    ]
    const { images, pdfs, others } = splitByKind(files)
    expect(images.map((f) => f.name)).toEqual(['a.png', 'd.jpg'])
    expect(pdfs.map((f) => f.name)).toEqual(['b.pdf'])
    expect(others.map((f) => f.name)).toEqual(['c.zip'])
  })

  it('★ type 为空串但扩展名是 .pdf 时仍算 PDF（只看 MIME 会误拒）', () => {
    const { pdfs, others } = splitByKind([{ type: '', name: '招标文件.PDF' }])
    expect(pdfs).toHaveLength(1)
    expect(others).toHaveLength(0)
  })

  it('type 缺失（undefined）不炸', () => {
    const { others } = splitByKind([{ type: undefined as unknown as string, name: 'x' }])
    expect(others).toHaveLength(1)
  })

  it('大小写混写的 MIME 也认', () => {
    const { images } = splitByKind([{ type: 'IMAGE/PNG', name: 'a' }])
    expect(images).toHaveLength(1)
  })

  it('空输入安全', () => {
    expect(splitByKind([])).toEqual({ images: [], pdfs: [], others: [] })
  })
})

describe('mergeImageAnalyses', () => {
  it('单张图**不加编号**（与改动前的格式逐字一致）', () => {
    const text = mergeImageAnalyses([{ name: 'a.png', analysis: '项目编号 A-1' }])
    expect(text).toBe('【用户上传图片的解析结果】\n项目编号 A-1')
  })

  it('★ 多张图逐张标注来源（否则模型分不清字段来自哪张）', () => {
    const text = mergeImageAnalyses([
      { name: '公告.png', analysis: '项目编号 A-1' },
      { name: '结果.png', analysis: '中标人：某某公司' },
    ])
    expect(text).toContain('〔图片 1：公告.png〕')
    expect(text).toContain('〔图片 2：结果.png〕')
    // 顺序必须与用户挂载顺序一致
    expect(text.indexOf('公告.png')).toBeLessThan(text.indexOf('结果.png'))
  })

  it('解析为空的图不占编号（避免"图片 2"其实是空的）', () => {
    const text = mergeImageAnalyses([
      { name: 'ok.png', analysis: '有内容' },
      { name: 'empty.png', analysis: '   ' },
    ])
    expect(text).toBe('【用户上传图片的解析结果】\n有内容')
  })

  it('全部为空时返回空串（调用方据此报错，而不是送一段空上下文）', () => {
    expect(mergeImageAnalyses([{ name: 'a', analysis: '' }])).toBe('')
    expect(mergeImageAnalyses([])).toBe('')
  })

  it('去掉解析文本首尾空白（后端返回常带换行）', () => {
    const text = mergeImageAnalyses([{ name: 'a', analysis: '\n\n内容\n\n' }])
    expect(text.endsWith('内容')).toBe(true)
  })

  it('缺名字时用占位而不是 undefined', () => {
    const text = mergeImageAnalyses([
      { name: '', analysis: 'x' },
      { name: 'b', analysis: 'y' },
    ])
    expect(text).toContain('〔图片 1：未命名〕')
    expect(text).not.toContain('undefined')
  })
})

describe('describeAdditions', () => {
  it('说明加载了几张、省了多少', () => {
    const plan = planAdditions(0, [img(1000)])
    const { text, kind } = describeAdditions(plan, [
      { originalBytes: 300 * 1024, bytes: 100 * 1024, compressed: true },
    ])
    expect(text).toContain('已加载 1 张图片')
    expect(text).toContain('200 KB')
    expect(kind).toBe('success')
  })

  it('★ 被拒时必须说清原因（否则用户只看到"没反应"）', () => {
    const plan = planAdditions(MAX_ATTACHMENTS, [img(1000), img(MAX_IMAGE_BYTES + 1)])
    const { text } = describeAdditions(plan, [])
    expect(text).toContain(`${MAX_ATTACHMENTS} 张`)
    expect(text).toContain('MB')
  })

  it('一张都没加成时用 warning 级提示', () => {
    const plan = planAdditions(MAX_ATTACHMENTS, [img(1000)])
    expect(describeAdditions(plan, []).kind).toBe('warning')
  })

  it('部分成功时仍是 success（不吓用户）', () => {
    const plan = planAdditions(MAX_ATTACHMENTS - 1, [img(1000), img(1000)])
    const { kind, text } = describeAdditions(plan, [
      { originalBytes: 1000, bytes: 1000, compressed: false },
    ])
    expect(kind).toBe('success')
    expect(text).toContain('多出的已跳过')
  })
})
