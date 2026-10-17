/**
 * 多图附件的**纯逻辑**：数量/体积闸门、解析结果合并、提示文案。
 *
 * ★ 为什么抽出来（而不是写在 Chat.tsx 里）
 * ====================================
 * 多图上传的三处规则都是"写错了不会报错、只会悄悄出错"的类型：
 *
 *   ① **闸门**：最多几张、总共能有多大。写错的后果是用户传了 10 张、
 *      界面卡死或后端 413，而**开发时通常只测一两张**
 *   ② **合并**：多张图的解析结果要拼成一段送进 Agent 的文本。拼错的后果是
 *      "第二张图的结论被当成第一张的"——模型不会报错，它只会答错
 *   ③ **提示文案**：被拒的图必须**说清为什么**（超张数 / 太大 / 不是图片），
 *      否则用户只会看到"没反应"
 *
 * 抽成纯函数后，这三件事都能用单测钉住，不必靠手动传图去试。
 */

/** 一次最多挂几张图。 */
export const MAX_ATTACHMENTS = 4

/**
 * 单张图的原始体积上限（压缩**前**）。
 *
 * 取 20MB：手机直出照片常在 3~8MB，20MB 能覆盖绝大多数截图与相册图；
 * 再大的多半是扫描件或 RAW，压缩到 1568px 前先在浏览器里读进内存会明显卡顿。
 * （压缩后的上限由 `lib/image.ts` 的 MAX_BASE64_CHARS 把关。）
 */
export const MAX_IMAGE_BYTES = 20 * 1024 * 1024

export interface AdditionPlan {
  /** 允许加入的下标（对应传入数组） */
  accepted: number[]
  /** 被拒的下标 → 原因（用于给用户一句人话） */
  rejected: { index: number; reason: 'too_many' | 'too_large' | 'not_image' }[]
}

/** PDF 的 MIME。拖拽进来的 `.pdf` 有时 type 为空串，故还要看扩展名。 */
export const PDF_MIME = 'application/pdf'

export interface SplitFiles<T> {
  images: T[]
  pdfs: T[]
  others: T[]
}

/**
 * 按类型分流：图片 / PDF / 其它。
 *
 * ★ 为什么扩展名也要认（`.pdf`）
 * ---------------------------
 * 从某些来源（旧浏览器、压缩包解出的文件、资源管理器拖拽）拿到的
 * `File.type` 是**空串**。只看 MIME 会把 PDF 误判成"非图片"直接拒掉，
 * 而用户明明拖的是 PDF —— 这种"什么都没发生"的反馈最难排查。
 */
export function splitByKind<T extends { type: string; name?: string }>(
  files: T[],
): SplitFiles<T> {
  const out: SplitFiles<T> = { images: [], pdfs: [], others: [] }
  for (const file of files) {
    const type = (file.type || '').toLowerCase()
    const name = (file.name || '').toLowerCase()
    if (type.startsWith('image/')) out.images.push(file)
    else if (type === PDF_MIME || name.endsWith('.pdf')) out.pdfs.push(file)
    else out.others.push(file)
  }
  return out
}

/**
 * 决定这一批文件里哪些能挂上去。
 *
 * 按输入顺序处理，**先到先得**：名额满了之后的都算 `too_many`。
 * 之所以不做"挑体积小的"之类的优化：用户拖进来 5 张、只留 4 张时，
 * 让他自己决定删哪张比我们替他挑更符合预期（我们会明确提示满了几张）。
 */
export function planAdditions(
  currentCount: number,
  files: { type: string; size: number }[],
  options: { maxCount?: number; maxBytes?: number } = {},
): AdditionPlan {
  const maxCount = options.maxCount ?? MAX_ATTACHMENTS
  const maxBytes = options.maxBytes ?? MAX_IMAGE_BYTES
  const plan: AdditionPlan = { accepted: [], rejected: [] }
  let used = Math.max(0, currentCount)

  files.forEach((file, index) => {
    // 顺序即优先级：先判类型（比"太大"更根本），再判体积，最后判名额
    if (!file.type || !file.type.startsWith('image/')) {
      plan.rejected.push({ index, reason: 'not_image' })
      return
    }
    if (file.size > maxBytes) {
      plan.rejected.push({ index, reason: 'too_large' })
      return
    }
    if (used >= maxCount) {
      plan.rejected.push({ index, reason: 'too_many' })
      return
    }
    plan.accepted.push(index)
    used += 1
  })
  return plan
}

export interface AnalysisEntry {
  /** 图片名（用于在合并文本里标注"哪张图"） */
  name: string
  /** 该图的解析结果；空串表示这张图没解析出内容 */
  analysis: string
}

/**
 * 把多张图的解析结果合并成一段送进 Agent 的问题文本。
 *
 * 格式（**这是与后端提示词之间的约定**，改动要同步看这里的用例）：
 *
 *     【用户上传图片的解析结果】
 *     〔图片 1：a.png〕
 *     <解析文本>
 *
 *     〔图片 2：b.png〕
 *     <解析文本>
 *
 * ★ 为什么必须逐张标注来源
 * ----------------------
 * 拼成一坨时，模型无法区分"哪些字段来自哪张图"。实测场景：用户先传招标公告
 * 截图、再传中标结果截图，两张图里都有"项目编号"——不标注就会出现
 * "用第二张的编号去核对第一张的金额"这类**看起来合理但完全错位**的结论。
 *
 * 单张图时**不加编号**：与改动前的格式逐字一致，避免历史会话/既有断言漂移。
 */
export function mergeImageAnalyses(entries: AnalysisEntry[]): string {
  const usable = entries.filter((e) => (e.analysis || '').trim())
  if (!usable.length) return ''
  if (usable.length === 1) {
    return `【用户上传图片的解析结果】\n${usable[0].analysis.trim()}`
  }
  const blocks = usable.map(
    (e, i) => `〔图片 ${i + 1}：${e.name || '未命名'}〕\n${e.analysis.trim()}`,
  )
  return `【用户上传图片的解析结果】\n${blocks.join('\n\n')}`
}

/** 附件条的提示文案（含被拒原因与压缩结果），供 toast 使用。 */
export function describeAdditions(
  plan: AdditionPlan,
  prepared: { originalBytes: number; bytes: number; compressed: boolean }[],
): { text: string; kind: 'success' | 'warning' } {
  const parts: string[] = []
  if (prepared.length) {
    const saved = prepared.reduce(
      (sum, p) => sum + Math.max(0, p.originalBytes - p.bytes),
      0,
    )
    parts.push(
      prepared.length === 1
        ? saved > 0
          ? `已加载 1 张图片（省 ${Math.round(saved / 1024)} KB）`
          : '已加载 1 张图片'
        : `已加载 ${prepared.length} 张图片${saved > 0 ? `（省 ${Math.round(saved / 1024)} KB）` : ''}`,
    )
  }
  const reasons = new Set(plan.rejected.map((r) => r.reason))
  if (reasons.has('too_many')) parts.push(`最多 ${MAX_ATTACHMENTS} 张，多出的已跳过`)
  if (reasons.has('too_large')) {
    parts.push(`超过 ${Math.round(MAX_IMAGE_BYTES / 1024 / 1024)}MB 的图片已跳过`)
  }
  if (reasons.has('not_image')) parts.push('非图片文件已跳过')

  return {
    text: parts.join('；') || '没有可添加的图片',
    kind: plan.rejected.length && !plan.accepted.length ? 'warning' : 'success',
  }
}
