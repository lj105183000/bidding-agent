/**
 * 上传图片的前端预处理：等比缩放 + 转 JPEG。
 *
 * 为什么必须压缩
 * --------------
 * 1. **绕开后端上限**：`/api/vision` 限制 base64 长度 4,000,000 字符（≈3MB 二进制）。
 *    手机拍的照片动辄 4~8MB，直接上传必定 413。
 * 2. **省流量与时间**：base64 会让体积再涨 ~33%，压缩后通常从 5MB 降到 200KB 以内，
 *    上传与模型推理都更快。
 * 3. **视觉模型不需要原始分辨率**：官方建议长边不超过 1568px，再大也不会提升识别率。
 *
 * 设计约束
 * --------
 * - 输出 **JPEG**：兼容性最好，且比 PNG 小得多（截图里的文字也够清晰）。
 *   注意 PNG 截图转 JPEG 后文字边缘会有轻微压缩痕迹，故质量给到 0.85。
 * - 若图片本身小于阈值，**不压缩**（避免无谓的重编码损失）。
 * - 若压缩后仍超限，逐档降质量重试。
 */

/** 后端 /api/vision 的限制（字符数） */
const MAX_BASE64_CHARS = 4_000_000

/** 长边上限：与视觉模型推荐值对齐 */
const MAX_EDGE = 1568

/** 小于这个体积（字节）就跳过压缩，避免无谓重编码 */
const SKIP_COMPRESS_BYTES = 300 * 1024

/** 质量降档序列：第一档清晰，后续逐档减小以兜住超大图 */
const QUALITY_STEPS = [0.85, 0.7, 0.55, 0.4]

export interface PreparedImage {
  /** `data:image/jpeg;base64,...` 形式，可直接交给后端 */
  dataUrl: string
  /** 用于 <img src> 的预览地址（与 dataUrl 相同，单独命名以表意） */
  previewUrl: string
  /** 压缩后的字节数（估算自 base64 长度） */
  bytes: number
  /** 原始文件字节数 */
  originalBytes: number
  /** 原始文件名 */
  name: string
  /** 是否实际做过压缩 */
  compressed: boolean
}

export class ImagePrepareError extends Error {}

/** base64 字符数 → 二进制字节数（含 padding 修正） */
function base64Bytes(dataUrl: string): number {
  const body = dataUrl.slice(dataUrl.indexOf(',') + 1)
  const padding = body.endsWith('==') ? 2 : body.endsWith('=') ? 1 : 0
  return Math.floor((body.length * 3) / 4) - padding
}

function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      URL.revokeObjectURL(url)
      resolve(img)
    }
    img.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new ImagePrepareError('无法读取该图片（文件可能已损坏或格式不受支持）'))
    }
    img.src = url
  })
}

/**
 * 把用户选的文件转成可上传的 data URL。
 *
 * 非图片文件、动图（GIF）会明确报错而不是静默失败——
 * GIF 只取首帧会让用户以为"识别不准"，不如直接说清楚。
 */
export async function prepareImage(file: File): Promise<PreparedImage> {
  if (!file.type.startsWith('image/')) {
    throw new ImagePrepareError(`「${file.name}」不是图片文件`)
  }
  if (file.type === 'image/gif') {
    throw new ImagePrepareError('暂不支持 GIF，请上传静态图片（JPG / PNG / WebP）')
  }

  const img = await loadImage(file)
  const originalBytes = file.size

  // 原图本身够小且尺寸不超标：直接读成 data URL，不重编码
  const longEdge = Math.max(img.naturalWidth, img.naturalHeight)
  if (originalBytes <= SKIP_COMPRESS_BYTES && longEdge <= MAX_EDGE) {
    const dataUrl = await readAsDataUrl(file)
    if (dataUrl.length <= MAX_BASE64_CHARS) {
      return {
        dataUrl,
        previewUrl: dataUrl,
        bytes: originalBytes,
        originalBytes,
        name: file.name,
        compressed: false,
      }
    }
  }

  // 等比缩放到长边上限（只缩不放，避免把小图拉大）
  const scale = longEdge > MAX_EDGE ? MAX_EDGE / longEdge : 1
  const targetW = Math.max(1, Math.round(img.naturalWidth * scale))
  const targetH = Math.max(1, Math.round(img.naturalHeight * scale))

  const canvas = document.createElement('canvas')
  canvas.width = targetW
  canvas.height = targetH
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    throw new ImagePrepareError('当前浏览器不支持图片压缩（canvas 不可用）')
  }
  // 白底：PNG 透明区域转 JPEG 会变黑，先铺白
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, targetW, targetH)
  ctx.drawImage(img, 0, 0, targetW, targetH)

  let dataUrl = ''
  for (const quality of QUALITY_STEPS) {
    dataUrl = canvas.toDataURL('image/jpeg', quality)
    if (dataUrl.length <= MAX_BASE64_CHARS) break
  }

  if (!dataUrl) {
    throw new ImagePrepareError('图片处理失败，请换一张试试')
  }
  if (dataUrl.length > MAX_BASE64_CHARS) {
    throw new ImagePrepareError(
      `图片过大（压缩后仍超 ${Math.round(MAX_BASE64_CHARS / 1_000_000)}MB 上限），请先裁剪或缩小后再上传`,
    )
  }

  return {
    dataUrl,
    previewUrl: dataUrl,
    bytes: base64Bytes(dataUrl),
    originalBytes,
    name: file.name,
    compressed: true,
  }
}

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result ?? ''))
    reader.onerror = () => reject(new ImagePrepareError('读取文件失败'))
    reader.readAsDataURL(file)
  })
}

/** 人类可读的体积文案（用于预览条上的"1.2 MB → 180 KB"） */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
