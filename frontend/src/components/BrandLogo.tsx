import logoSource from '../../public/logo.svg?raw'

/**
 * 品牌 Logo（招采智脑）。
 *
 * ★ 为什么用 `?raw` 导入 SVG 源码，而不是内联复制一份 markup
 * -------------------------------------------------
 * 同一份图形要在**两处**使用，且必须**逐像素一致**：
 *    · `public/logo.svg` —— favicon / 直接访问（HTML `<link rel="icon">`）
 *    · 侧栏与移动端顶栏 —— 页面内的品牌标识
 *
 * 若在组件里另写一份内联 SVG，两处就会各自演化：
 * 改了一处忘了另一处，favicon 与页面 logo 不一致 —— 而这种不一致
 * **很难被发现**（两者从不同时出现在同一屏，且 favicon 只有 16px）。
 * 故让页面组件也读**同一个文件**，从根上消除这个风险。
 *
 * 代价：多一次 `?raw` 导入（构建期内联成字符串，运行时零开销）。
 *
 * ★ 为什么不直接用 `<img src="/logo.svg">`
 * ------------------------------------
 * `<img>` 渲染的 SVG 是**独立文档**，外部 CSS 无法作用于其内部元素。
 * 本项目需要在浅色/深色两种底色下都保证可辨识，保留"可被 CSS 微调"的
 * 能力比为省几行代码更值。
 *
 * ★ 安全说明（为什么 dangerouslySetInnerHTML 在这里是可接受的）
 * ----------------------------------------------------
 * 内容是**构建期内联的本地静态文件**（`public/logo.svg`），
 * 不含任何用户输入、不来自网络、无插值。Vite 在构建时把它变成字符串常量，
 * 运行时不存在注入路径。
 */

/** 让导入的 SVG 自适应容器尺寸。
 *
 * `logo.svg` 自带 `width="64" height="64"`（favicon 需要固定尺寸），
 * 但页面内要靠 Tailwind 的 `h-8 w-8` 控制大小。若不移除这两个属性，
 * CSS 虽然通常能覆盖，但**在 SVG 上依赖 CSS 覆盖 width/height 是不稳的**
 * （部分环境里 SVG 的呈现属性优先级表现不一致）。
 * 故这里显式改写为 `100%`，让尺寸**只**由 CSS 决定，行为可预期。
 *
 * 同时清掉 `aria-label`：无障碍语义由外层容器的 `role="img"` 提供，
 * 内联后若保留会与外层重复，屏幕阅读器可能读两遍。
 */
function prepareForInline(source: string): string {
  return (
    source
      // 注释会让 dangerouslySetInnerHTML 的内容变长，且对渲染无用
      .replace(/<!--[\s\S]*?-->/g, '')
      .replace(
        /<svg\b[^>]*>/,
        (tag) =>
          tag
            .replace(/\swidth="[^"]*"/, ' width="100%"')
            .replace(/\sheight="[^"]*"/, ' height="100%"')
            .replace(/\srole="[^"]*"/, '')
            .replace(/\saria-label="[^"]*"/, '')
            .replace(
              /<svg\b/,
              '<svg preserveAspectRatio="xMidYMid meet" focusable="false"',
            ),
      )
      .trim()
  )
}

const INLINE_LOGO = prepareForInline(logoSource)

export default function BrandLogo({ className = '' }: { className?: string }) {
  return (
    <span
      role="img"
      aria-label="招采智脑"
      className={`inline-flex shrink-0 items-center justify-center ${className}`}
      // 见文件头「安全说明」：内容是本地静态文件、构建期内联、无插值
      dangerouslySetInnerHTML={{ __html: INLINE_LOGO }}
    />
  )
}
