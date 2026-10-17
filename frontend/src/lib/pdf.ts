/**
 * PDF → 图片（**前端转换，后端仍只收图片**）。
 *
 * ★ 为什么在前端转（而不是让后端接 PDF）
 * ==================================
 * 后端的视觉链路只认图片（`/api/vision` 收 base64 图片）。若在后端加 PDF 解析，
 * 要引入服务端渲染/栅格化依赖（pdfium / mupdf / 需系统库），部署面明显变重；
 * 而浏览器**自带** PDF 渲染能力，转换成本落在客户端，后端一行不用改。
 *
 * ★ 为什么必须**动态 import**（首屏体积）
 * ===================================
 * `pdfjs-dist` 的浏览器主包 ~448KB（min）、worker 另 ~1.2MB。本项目的首屏
 * 已经被压到 **245.97 kB**（路由级懒加载的成果，见 `App.tsx` 的说明）——
 * 静态 import 会让首屏直接翻到 ~700KB，而**绝大多数用户根本不会传 PDF**。
 * 故这里用 `await import(...)`：Vite 会把它切成独立 chunk，
 * 只有真的拖进 PDF 时才下载。
 *
 * ★ 转换参数怎么定的
 * ==============
 * 长边缩放到 `TARGET_EDGE`（1568）：与 `lib/image.ts` 的 `MAX_EDGE` 对齐，
 * 这样转出来的图**不需要二次压缩**就能过视觉模型（少一次重编码，图也更清晰）。
 * 上限 2 倍：小页面的 PDF（如 A5）不必强行放大，放大只增体积不增信息。
 * 输出 JPEG(0.9)：扫描件类的 PDF 用 JPEG 比 PNG 小得多，且视觉模型对
 * JPEG 压缩不敏感。
 *
 * ⚠️ 已知取舍（如实记录）
 * ==================
 * **转图会丢掉文字层**：电子版 PDF 里的可选文本本是"免费且零误差"的信息，
 * 栅格化后要交给视觉模型重新识别，精度必然低于直接抽文本。
 * 故这条路径适合**扫描件 / 图片型 PDF / 表格截图**；若将来 PDF 成为主要输入，
 * 更值得做的是后端加一条"PDF 抽文本"路径（pdfminer 之类），而不是继续转图。
 */

/** 一次最多转换几页。 */
export const MAX_PDF_PAGES = 5

/** 渲染长边目标像素（与 `lib/image.ts` 的 MAX_EDGE 对齐，避免二次压缩）。 */
const TARGET_EDGE = 1568

/** 渲染缩放上限：小页面不必强行放大（放大只增体积，不增信息）。 */
const MAX_SCALE = 3

/** 单页渲染超时（毫秒）。损坏/异常复杂的 PDF 可能让 render 永远不 resolve。 */
const RENDER_TIMEOUT_MS = 20_000

export interface PdfConvertResult {
  /** 逐页转出的图片文件（可直接交给 `prepareImage`） */
  files: File[]
  /** PDF 总页数 */
  pageCount: number
  /** 实际转换成功的页数 */
  converted: number
  /** 因超过页数上限而未转换的页数 */
  skipped: number
  /** 单页转换失败的数量（不阻断其余页） */
  failed: number
}

export class PdfConvertError extends Error {}

let workerReady = false

/**
 * 装好 pdfjs 的 worker。
 *
 * ★ 为什么必须显式指定 workerSrc
 * ---------------------------
 * pdfjs 默认会去猜 worker 路径，在打包后的产物里几乎必然猜错（404），
 * 结果是**主线程解析**（大 PDF 会把界面卡死几秒）或直接报错。
 * 用 Vite 的 `?url` 让打包器产出该文件并给出真实 URL。
 */
async function loadPdfjs() {
  const pdfjs = await import('pdfjs-dist')
  if (!workerReady) {
    const workerUrl = (await import('pdfjs-dist/build/pdf.worker.min.mjs?url')).default
    pdfjs.GlobalWorkerOptions.workerSrc = workerUrl
    workerReady = true
  }
  return pdfjs
}

/** 把文件名换成"去掉扩展名"的形式（用于给每页生成名字）。 */
function baseName(name: string): string {
  return (name || 'pdf').replace(/\.pdf$/i, '') || 'pdf'
}

function withTimeout<T>(task: Promise<T>, ms: number, what: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new PdfConvertError(`${what}超时（超过 ${Math.round(ms / 1000)} 秒）`)),
      ms,
    )
    task.then(
      (value) => {
        clearTimeout(timer)
        resolve(value)
      },
      (err) => {
        clearTimeout(timer)
        reject(err)
      },
    )
  })
}

export async function pdfToImages(
  file: File,
  options: { maxPages?: number } = {},
): Promise<PdfConvertResult> {
  const maxPages = options.maxPages ?? MAX_PDF_PAGES
  let pdfjs: Awaited<ReturnType<typeof loadPdfjs>>
  try {
    pdfjs = await loadPdfjs()
  } catch (err) {
    throw new PdfConvertError(
      `PDF 解析组件加载失败：${err instanceof Error ? err.message : String(err)}`,
    )
  }

  let doc
  let task
  try {
    const data = new Uint8Array(await file.arrayBuffer())
    // 保留 loading task 引用：释放资源要调 `task.destroy()`
    // （6.x 的类型里 PDFDocumentProxy 没有 destroy，只有 cleanup）
    task = pdfjs.getDocument({ data })
    doc = await task.promise
  } catch (err) {
    // 加密 / 损坏分别给一句人话：这两种的处理方式完全不同
    const name = (err as { name?: string })?.name ?? ''
    if (name === 'PasswordException') {
      throw new PdfConvertError('这个 PDF 有密码保护，请先解除密码后再上传')
    }
    if (name === 'InvalidPDFException') {
      throw new PdfConvertError('这个 PDF 无法解析（文件可能已损坏）')
    }
    throw new PdfConvertError(
      `PDF 打开失败：${err instanceof Error ? err.message : String(err)}`,
    )
  }

  const pageCount = doc.numPages
  const todo = Math.min(pageCount, Math.max(1, maxPages))
  const files: File[] = []
  let failed = 0

  try {
    for (let pageNo = 1; pageNo <= todo; pageNo += 1) {
      try {
        const page = await doc.getPage(pageNo)
        const base = page.getViewport({ scale: 1 })
        const longEdge = Math.max(base.width, base.height) || 1
        const scale = Math.min(MAX_SCALE, TARGET_EDGE / longEdge)
        const viewport = page.getViewport({ scale })

        const canvas = document.createElement('canvas')
        canvas.width = Math.max(1, Math.round(viewport.width))
        canvas.height = Math.max(1, Math.round(viewport.height))
        const ctx = canvas.getContext('2d')
        if (!ctx) throw new PdfConvertError('浏览器不支持 canvas，无法转换 PDF')
        // 白底：PDF 透明背景渲染出来是黑的（视觉模型会把它当"深色图片"）
        ctx.fillStyle = '#ffffff'
        ctx.fillRect(0, 0, canvas.width, canvas.height)

        await withTimeout(
          page.render({ canvas, canvasContext: ctx, viewport }).promise,
          RENDER_TIMEOUT_MS,
          `第 ${pageNo} 页渲染`,
        )
        page.cleanup()

        const blob = await new Promise<Blob | null>((resolve) =>
          canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.9),
        )
        if (!blob) throw new PdfConvertError(`第 ${pageNo} 页导出图片失败`)
        files.push(
          new File([blob], `${baseName(file.name)}-p${pageNo}.jpg`, {
            type: 'image/jpeg',
            lastModified: Date.now(),
          }),
        )
      } catch (err) {
        // 单页失败不放弃整份文档：多页 PDF 里坏一页很常见
        failed += 1
        if (err instanceof PdfConvertError && pageNo === 1 && files.length === 0) {
          // 第一页就失败 → 大概率整份文档不可用，直接抛（否则用户拿到的是莫名的"空结果"）
          throw err
        }
      }
    }
  } finally {
    // 释放 worker 侧资源；即使中途抛异常也要走。
    // 用 `task.destroy()`：它会连 worker 文档实例一起释放
    // （只调 `doc.cleanup()` 会留下缓存的页面位图）。
    try {
      await task?.destroy()
    } catch {
      /* 释放失败不影响结果 */
    }
  }

  return {
    files,
    pageCount,
    converted: files.length,
    skipped: Math.max(0, pageCount - todo),
    failed,
  }
}
